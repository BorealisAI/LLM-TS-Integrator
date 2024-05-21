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
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from utils.tools import (
    dataset2description,
    process_sample,
    get_next_batch
)
from utils.cka import linear_CKA, kernel_CKA

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

import pandas
from models.MutualInfo import MutualInfo
from models.WeightNet import WNet
import higher
import random

import os

warnings.filterwarnings("ignore")


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.hidden_size = None
        if os.path.exists(self.args.root_path + "/feature_data.pt"):
            self.llm_emb = (
                torch.load(self.args.root_path + "/feature_data.pt")
                .to(self.device)
                .detach()
            )

    def _build_model(self):
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
        criterion = nn.MSELoss(reduce=False)
        return criterion

    def _mutual_information(self, mutualinfo_model, indexs, prob_mutual):
        train_data, train_loader = self._get_data(flag="train")
        train_d = train_loader.dataset.train
        N = train_loader.dataset.__len__()
        step = train_loader.dataset.step
        win_size = train_loader.dataset.win_size

        indexs = torch.LongTensor(indexs).cpu().numpy().tolist()
        limit = self.llm_emb.shape[0] - 1
        indexs = [min(x, limit) for x in indexs]

        model_emb = []
        llm_emb = self.llm_emb[indexs]
        for i in range(len(indexs)):
            index = indexs[i]
            tmp = np.float32(train_d[index : index + win_size])
            sample = (
                torch.from_numpy(tmp)
                .reshape(1, tmp.shape[0], -1)
                .float()
                .to(self.device)
            )
            sample_emb = self.model.anomaly_detection(sample, return_hidden=True)
            model_emb.append(sample_emb.mean(1).reshape(1, -1))
        model_emb = torch.cat(model_emb, dim=0).to(self.device)
        mutualinfo_estimation = mutualinfo_model(llm_emb, model_emb, prob_mutual)
        return mutualinfo_estimation

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, indexs) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = torch.mean(criterion(pred, true))
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def pretrain_model(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

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
            for i, (batch_x, batch_y, indexs) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                loss = torch.mean(criterion(outputs, batch_x))
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
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        self.test(setting, test=1)
        return self.model

    def compute_cka(self):
        print("compute_cka")
        train_data, train_loader = self._get_data(flag="train")
        train_d = train_loader.dataset.train
        N = min(train_loader.dataset.__len__(), 5000)
        step = train_loader.dataset.step
        win_size = train_loader.dataset.win_size

        first_layers = []
        last_layers = []
        self.model.eval()
        for index in range(N):
            index = index * step
            sample = (
                torch.from_numpy(
                    np.float32(train_d[index : index + win_size]).reshape(
                        1, win_size, -1
                    )
                )
                .float()
                .to(self.device)
            )
            with torch.no_grad():
                first_layer, last_layer = self.model.anomaly_detection(
                    sample, return_hidden=False, return_first=True
                )
                first_layer, last_layer = (
                    first_layer.detach().data,
                    last_layer.detach().data,
                )
            first_layers.append(first_layer.reshape(1, -1))
            last_layers.append(last_layer.reshape(1, -1))
        first_layers = torch.cat(first_layers, dim=0).cpu().numpy()
        last_layers = torch.cat(last_layers, dim=0).cpu().numpy()
        print("linear_CKA", linear_CKA(first_layers, last_layers))
        print("kernel_CKA", kernel_CKA(first_layers, last_layers))

    def pretrain_mutual(self, path, mutualinfo=None):
        train_data, train_loader = self._get_data(flag="train")
        train_d = train_loader.dataset.train
        N = min(train_loader.dataset.__len__(), 10000)
        step = train_loader.dataset.step
        win_size = train_loader.dataset.win_size

        model_emb = []
        self.model.eval()
        for index in range(N):
            index = index * step
            sample = (
                torch.from_numpy(
                    np.float32(train_d[index : index + win_size]).reshape(
                        1, win_size, -1
                    )
                )
                .float()
                .to(self.device)
            )
            with torch.no_grad():
                sample_emb = self.model.anomaly_detection(
                    sample, return_hidden=True
                ).data
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
        llm_emb = self.llm_emb[0:N]
        val_len = min(int(N * 0.3), 5000)
        llm_emb_val, llm_emb_train = llm_emb[0:val_len, :], llm_emb[val_len:, :]
        model_emb_val, model_emb_train = model_emb[0:val_len, :], model_emb[val_len:, :]
        best_val_loss = -mutualinfo(llm_emb_val, model_emb_val)  # float('inf')
        mutual_path = os.path.join(path, "mutualinfo.pt")
        assert (
            llm_emb_train.shape[0] == model_emb_train.shape[0]
        ), "llm_emb_train shape[0] not equal to model_emb_train.shape[0]"
        for i in range(T):
            batch_size = min(int(N * 0.2), 5000)
            batch_index = random.sample(range(llm_emb_train.shape[0]), batch_size)
            loss = -mutualinfo(llm_emb_train[batch_index], model_emb_train[batch_index])
            opt.zero_grad()
            loss.backward()
            opt.step()
            val_loss = -mutualinfo(llm_emb_val, model_emb_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
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
            for i, (batch_x, batch_y, indexs) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                with higher.innerloop_ctx(self.model, model_optim) as (
                    fmodel,
                    diffopt,
                ):
                    # build the inner connection
                    outputs = fmodel(batch_x, None, None, None)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, :, f_dim:]
                    loss_value = torch.mean(
                        criterion(outputs, batch_x), dim=(1, 2)
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
                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                loss_value = torch.mean(
                    criterion(outputs, batch_x), dim=(1, 2)
                ).reshape(-1, 1)

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
                model_optim.step()
            mutualinfo_model = self.pretrain_mutual(path_prt, mutualinfo_model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        self.test(setting, test=1)
        return self.model

    def vali_loss(self, batch_data, criterion, fmodel):
        batch_x, batch_y, indexs = batch_data
        batch_x = batch_x.float().to(self.device)
        outputs = fmodel(batch_x, None, None, None)
        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, :, f_dim:]
        loss_value = torch.mean(criterion(outputs, batch_x).reshape(-1, 1))
        return loss_value

    def test(self, setting, test=0):
        if self.args.model == "TimesNet2":
            self.compute_cka()
        test_data, test_loader = self._get_data(flag="test")
        train_data, train_loader = self._get_data(flag="train")

        attens_energy = []
        folder_path = setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y, indexs) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y, indexs) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )

        f = open("result_anomaly_detection.txt", "a")
        f.write(setting + "  \n")
        f.write(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return