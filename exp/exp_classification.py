# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TimesNet (https://arxiv.org/abs/2210.02186) implementation
# from https://github.com/thuml/Time-Series-Library by THUML @ Tsinghua University
####################################################################################

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from utils.tools import (
    dataset2description,
    process_sample,
    get_next_batch,
    dataset2dimension,
)
from utils.cka import linear_CKA, kernel_CKA

import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from models.MutualInfo import MutualInfo
from models.WeightNet import WNet
import higher
import random

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.hidden_size = None
        if os.path.exists(self.args.root_path + "/feature_data.pt"):
            self.llm_emb = (
                torch.load(self.args.root_path + "/feature_data.pt")
                .to(self.device)
                .detach()
            )

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag="TRAIN")
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss(reduce=False)
        return criterion

    def instance_norm(self, case):
        if (
            self.args.root_path.count("EthanolConcentration") > 0
        ):  # special process for numerical stability
            case = torch.from_numpy(case)
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(
                torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            case /= stdev
            return case.numpy()
        else:
            return case

    def _mutual_information(self, mutualinfo_model, indexs, prob_mutual):
        target_seq_len = dataset2dimension(self.args.root_path.split("/")[-2])
        train_data, train_loader = self._get_data(flag="TRAIN")
        all_IDs = train_loader.dataset.all_IDs
        feature_df = train_loader.dataset.feature_df

        indexs = torch.LongTensor(indexs).cpu().numpy().tolist()
        limit = self.llm_emb.shape[0] - 1
        indexs = [min(x, limit) for x in indexs]

        N = len(indexs)
        model_emb = []
        llm_emb = self.llm_emb[indexs]
        for i in range(N):
            index = indexs[i]
            sample = np.expand_dims(
                self.instance_norm(feature_df.loc[all_IDs[index]].values), 0
            )
            sample, mask = self.pad_and_create_mask(sample, target_seq_len)
            sample = torch.from_numpy(sample).to(self.device).float()
            mask = torch.from_numpy(mask).to(self.device).float()
            sample_emb = self.model.classification(sample, mask, True)
            model_emb.append(sample_emb.mean(1).reshape(1, -1))
        model_emb = torch.cat(model_emb, dim=0).to(self.device)
        mutualinfo_estimation = mutualinfo_model(llm_emb, model_emb, prob_mutual)
        return mutualinfo_estimation

    def pad_and_create_mask(self, sample, target_seq_len=29):
        """
        Pad a numpy array to a target sequence length and create a mask for the original data.

        Parameters:
        - sample: A numpy array of shape (1, seq_len, D)
        - target_seq_len: The target sequence length to pad the array to

        Returns:
        - padded_sample: The padded numpy array of shape (1, target_seq_len, D)
        - mask: A mask array of shape (1, target_seq_len, D) where positions corresponding to
                the original data in sample are marked as 1, and padding positions as 0.
        """
        # Extract original dimensions
        _, seq_len, D = sample.shape

        # Check if padding is necessary
        if seq_len >= target_seq_len:
            return sample, np.ones_like(sample)

        # Calculate padding size
        padding_size = target_seq_len - seq_len

        # Create padding of zeros
        padding = np.zeros((1, padding_size, D))

        # Pad the original sample
        padded_sample = np.concatenate([sample, padding], axis=1)

        # Create mask (1 for original data, 0 for padding)
        mask = np.ones((padded_sample.shape[0], padded_sample.shape[1]))
        mask[:, seq_len:] = 0

        return padded_sample, mask

    def pretrain_model(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="TEST")
        test_data, test_loader = self._get_data(flag="TEST")

        path = os.path.join(self.args.checkpoints, setting + "_prt")
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask, indexs) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = torch.mean(criterion(outputs, label.long().squeeze(-1)))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    val_accuracy,
                    test_loss,
                    test_accuracy,
                )
            )
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        self.test(setting, test=1)
        return self.model

    def compute_cka(self):
        print("compute_cka")
        target_seq_len = dataset2dimension(self.args.root_path.split("/")[-2])
        train_data, train_loader = self._get_data(flag="TRAIN")
        all_IDs = train_loader.dataset.all_IDs
        feature_df = train_loader.dataset.feature_df
        N = min(len(all_IDs), 5000)

        # obtain model embedding and llm embedding
        first_layers = []
        last_layers = []
        self.model.eval()
        for index in range(N):
            sample = np.expand_dims(
                self.instance_norm(feature_df.loc[all_IDs[index]].values), 0
            )
            sample, mask = self.pad_and_create_mask(sample, target_seq_len)
            sample = torch.from_numpy(sample).to(self.device).float()
            mask = torch.from_numpy(mask).to(self.device).float()
            with torch.no_grad():
                first_layer, last_layer = self.model.classification(
                    sample, mask, return_hidden=False, return_first=True
                )
                first_layer, last_layer = (
                    first_layer.detach().data,
                    last_layer.detach().data,
                )
            first_layers.append(first_layer.reshape(1, -1))
            last_layers.append(last_layer.reshape(1, -1))
        first_layers = torch.cat(first_layers, dim=0).cpu().numpy()
        last_layers = torch.cat(last_layers, dim=0).cpu().numpy()
        print("shape", first_layers.shape, last_layers.shape)
        print("linear_CKA", linear_CKA(first_layers, last_layers))
        print("kernel_CKA", kernel_CKA(first_layers, last_layers))

    def pretrain_mutual(self, path, mutualinfo=None):
        target_seq_len = dataset2dimension(self.args.root_path.split("/")[-2])
        train_data, train_loader = self._get_data(flag="TRAIN")
        all_IDs = train_loader.dataset.all_IDs
        feature_df = train_loader.dataset.feature_df
        N = len(all_IDs)

        model_emb = []
        self.model.eval()
        for index in range(N):
            sample = np.expand_dims(
                self.instance_norm(feature_df.loc[all_IDs[index]].values), 0
            )
            sample, mask = self.pad_and_create_mask(sample, target_seq_len)
            sample = torch.from_numpy(sample).to(self.device).float()
            mask = torch.from_numpy(mask).to(self.device).float()
            with torch.no_grad():
                sample_emb = self.model.classification(sample, mask, True).detach()
            model_emb.append(sample_emb.mean(1).reshape(1, -1))
        model_emb = torch.cat(model_emb, dim=0).to(self.device)

        # load model
        if mutualinfo == None:
            self.hidden_size = model_emb.shape[1]
            mutualinfo = MutualInfo(input_emb_size=self.hidden_size).to(self.device)
            T = 1000
            print("mutual info train")
        else:
            T = 100
            print("mutual info finetune")
        opt = optim.Adam(mutualinfo.parameters(), lr=1e-3, weight_decay=1e-3)
        # model optimization
        val_len = int(N * 0.3)
        llm_emb_val, llm_emb_train = (
            self.llm_emb[0:val_len, :],
            self.llm_emb[val_len:, :],
        )
        model_emb_val, model_emb_train = model_emb[0:val_len, :], model_emb[val_len:, :]
        best_val_loss = -mutualinfo(llm_emb_val, model_emb_val)  # float('inf')
        mutual_path = os.path.join(path, "mutualinfo.pt")
        print(llm_emb_train.shape[0], model_emb_train.shape[0])
        assert (
            llm_emb_train.shape[0] == model_emb_train.shape[0]
        ), "llm_emb_train shape[0] not equal to model_emb_train.shape[0]"
        for i in range(T):
            batch_index = random.sample(range(llm_emb_train.shape[0]), int(N * 0.2))
            loss = -mutualinfo(llm_emb_train[batch_index], model_emb_train[batch_index])
            opt.zero_grad()
            loss.backward()
            opt.step()
            val_loss = -mutualinfo(llm_emb_val, model_emb_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("saving")
                print("iteration", i, "train loss", loss, "val loss", val_loss)
                torch.save(mutualinfo.state_dict(), mutual_path)
        mutualinfo.load_state_dict(torch.load(mutual_path))
        # valid
        batch_index1 = random.sample(range(llm_emb_train.shape[0]), int(N * 0.2))
        batch_index2 = random.sample(range(llm_emb_train.shape[0]), int(N * 0.2))
        m11 = mutualinfo(llm_emb_train[batch_index1], model_emb_train[batch_index1])
        m12 = mutualinfo(llm_emb_train[batch_index1], model_emb_train[batch_index2])
        m21 = mutualinfo(llm_emb_train[batch_index2], model_emb_train[batch_index1])
        m22 = mutualinfo(llm_emb_train[batch_index2], model_emb_train[batch_index2])
        print("mutual info", "m11", m11, "m12", m12, "m21", m21, "m22", m22)
        return mutualinfo

    def train(self, setting):
        # load pre-trained model
        path_prt = os.path.join(self.args.checkpoints, setting + "_prt")
        best_model_path = path_prt + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        print("pre-trained model loaded")

        # load pre-trained mutual info
        mutualinfo_model = self.pretrain_mutual(path_prt, None)
        print("mutualinfo_model loaded")

        # train
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="TEST")
        test_data, test_loader = self._get_data(flag="TEST")
        batch_generator = get_next_batch(vali_loader)
        # load WeightNet
        wnet = WNet().to(self.device)
        wnet_opt = torch.optim.Adam(wnet.parameters(), lr=0.01)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask, indexs) in enumerate(train_loader):
                indexs = list(indexs)

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                with higher.innerloop_ctx(self.model, model_optim) as (
                    fmodel,
                    diffopt,
                ):
                    # build the inner connection
                    outputs = fmodel(batch_x, padding_mask, None, None)
                    loss_value = criterion(
                        outputs, label.long().squeeze(-1)
                    ).reshape(-1, 1)
                    omega_sample, gamma_sample = wnet(loss_value)
                    prob_mutual = gamma_sample / (torch.sum(gamma_sample) + 1e-6)
                    mutual_info_loss = -self._mutual_information(
                            mutualinfo_model, indexs, prob_mutual
                        )
                    loss = (
                        torch.mean(omega_sample * loss_value)
                        + torch.mean(gamma_sample) * mutual_info_loss
                    )
                    fmodel.zero_grad()
                    diffopt.step(loss)
                    # compute the outer level loss
                    batch_data = next(batch_generator)
                    val_loss = self.vali_loss(batch_data, criterion, fmodel)
                    wnet_opt.zero_grad()
                    val_loss.backward()
                    wnet_opt.step()
                outputs = self.model(batch_x, padding_mask, None, None)
                loss_value = criterion(outputs, label.long().squeeze(-1)).reshape(-1, 1)
                omega_sample, gamma_sample = wnet(loss_value)
                omega_sample, gamma_sample = (
                    omega_sample.detach(),
                    gamma_sample.detach(),
                )
                prob_mutual = gamma_sample / (torch.sum(gamma_sample) + 1e-6)
                mutual_info_loss = -self._mutual_information(
                        mutualinfo_model, indexs, prob_mutual
                    )
                loss = (
                    torch.mean(omega_sample * loss_value)
                    + torch.mean(gamma_sample) * mutual_info_loss
                )
                if i % 5 == 0:
                    print(
                        "weight_sample after",
                        torch.mean(omega_sample),
                        torch.mean(gamma_sample),
                    )

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
            mutualinfo_model = self.pretrain_mutual(path_prt, mutualinfo_model)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    val_accuracy,
                    test_loss,
                    test_accuracy,
                )
            )
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        self.test(setting, test=1)
        return self.model

    def vali_loss(self, batch_data, criterion, fmodel):
        batch_x, label, padding_mask, indexs = batch_data
        batch_x = batch_x.float().to(self.device)
        padding_mask = padding_mask.float().to(self.device)
        label = label.to(self.device)
        outputs = fmodel(batch_x, padding_mask, None, None)
        loss = torch.mean(criterion(outputs, label.long().squeeze()))
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, indexs) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(torch.mean(loss))

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def test(self, setting, test=0):
        if self.args.model == "TimesNet2":
            self.compute_cka()
        test_data, test_loader = self._get_data(flag="TEST")

        trues = []
        folder_path = "test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, indexs) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print("test shape:", preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = "results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print("accuracy:{}".format(accuracy))
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write("accuracy:{}".format(accuracy))
        f.write("\n")
        f.write("\n")
        f.close()
        return
