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
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.tools import (
    dataset2description,
    process_sample,
    get_next_batch,
    get_gpu_memory,
)
from utils.metrics import metric
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


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.hidden_size = None
        if os.path.exists(self.args.root_path + "/feature_data.pt"):
            self.llm_emb = (
                torch.load(self.args.root_path + "/feature_data.pt").cpu().detach()
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
        data_x = train_loader.dataset.data_x
        data_stamp = train_loader.dataset.data_stamp
        seq_len = train_loader.dataset.seq_len

        indexs = torch.LongTensor(indexs).cpu().numpy().tolist()
        limit = self.llm_emb.shape[0] - 1
        indexs = [min(x, limit) for x in indexs]

        model_emb = []
        llm_emb = self.llm_emb[indexs]
        llm_emb = llm_emb.to(self.device)
        for i in range(len(indexs)):
            index = indexs[i]
            s_begin = index
            s_end = s_begin + seq_len
            seq_x = data_x[s_begin:s_end]
            seq_x_mark = data_stamp[s_begin:s_end]
            sample = (
                torch.from_numpy(
                    np.array(seq_x).reshape(1, seq_x.shape[0], seq_x.shape[1])
                )
                .float()
                .to(self.device)
            )
            seq_x_mark = (
                torch.from_numpy(
                    np.array(seq_x_mark).reshape(
                        1, seq_x_mark.shape[0], seq_x_mark.shape[1]
                    )
                )
                .float()
                .to(self.device)
            )
            sample_emb = self.model.forecast(
                sample, seq_x_mark, None, None, return_hidden=True
            )
            model_emb.append(sample_emb.mean(1).reshape(1, -1))
        model_emb = torch.cat(model_emb, dim=0).to(self.device)
        mutualinfo_estimation = mutualinfo_model(llm_emb, model_emb, prob_mutual)
        return mutualinfo_estimation

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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(get_gpu_memory())
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, indexs) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = torch.mean(criterion(outputs, batch_y))
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = torch.mean(criterion(outputs, batch_y))
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

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
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
            # break

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        self.test(setting, test=1)
        return self.model

    def compute_cka(self):
        print("compute_cka")
        train_data, train_loader = self._get_data(flag="train")
        data_x = train_loader.dataset.data_x
        data_stamp = train_loader.dataset.data_stamp
        seq_len = train_loader.dataset.seq_len
        N = min(train_loader.dataset.__len__(), 5000)

        first_layers = []
        last_layers = []
        self.model.eval()
        for index in range(N):
            s_begin = index
            s_end = s_begin + seq_len
            seq_x = data_x[s_begin:s_end]
            seq_x_mark = data_stamp[s_begin:s_end]
            sample = (
                torch.from_numpy(
                    np.array(seq_x).reshape(1, seq_x.shape[0], seq_x.shape[1])
                )
                .float()
                .to(self.device)
            )
            seq_x_mark = (
                torch.from_numpy(
                    np.array(seq_x_mark).reshape(
                        1, seq_x_mark.shape[0], seq_x_mark.shape[1]
                    )
                )
                .float()
                .to(self.device)
            )
            with torch.no_grad():
                first_layer, last_layer = self.model.forecast(
                    sample,
                    seq_x_mark,
                    None,
                    None,
                    return_hidden=False,
                    return_first=True,
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
        data_x = train_loader.dataset.data_x
        data_stamp = train_loader.dataset.data_stamp
        seq_len = train_loader.dataset.seq_len
        N = min(train_loader.dataset.__len__(), self.llm_emb.shape[0], 10000)

        model_emb = []
        self.model.eval()
        for index in range(N):
            s_begin = index
            s_end = s_begin + seq_len
            seq_x = data_x[s_begin:s_end]
            seq_x_mark = data_stamp[s_begin:s_end]
            sample = (
                torch.from_numpy(
                    np.array(seq_x).reshape(1, seq_x.shape[0], seq_x.shape[1])
                )
                .float()
                .to(self.device)
            )
            seq_x_mark = (
                torch.from_numpy(
                    np.array(seq_x_mark).reshape(
                        1, seq_x_mark.shape[0], seq_x_mark.shape[1]
                    )
                )
                .float()
                .to(self.device)
            )
            with torch.no_grad():
                sample_emb = self.model.forecast(
                    sample, seq_x_mark, None, None, return_hidden=True
                ).data
            model_emb.append(sample_emb.mean(1).reshape(1, -1))
        model_emb = torch.cat(model_emb, dim=0).to(self.device)
        # load model
        if mutualinfo == None:
            self.hidden_size = model_emb.shape[1]
            mutualinfo = MutualInfo(
                input_emb_size=self.hidden_size, llm_emb_size=self.llm_emb.shape[1]
            ).to(self.device)
            T = 1000
            print("mutual info train")
            opt = optim.Adam(mutualinfo.parameters(), lr=1e-3, weight_decay=1e-3)
        else:
            T = 100
            print("mutual info finetune")
            opt = optim.Adam(mutualinfo.parameters(), lr=1e-4, weight_decay=1e-3)
        #
        llm_emb = self.llm_emb[0:N]
        llm_emb = llm_emb.to(self.device)
        val_len = int(N * 0.3)
        llm_emb_val, llm_emb_train = llm_emb[0:val_len, :], llm_emb[val_len:, :]
        model_emb_val, model_emb_train = model_emb[0:val_len, :], model_emb[val_len:, :]
        best_val_loss = -mutualinfo(llm_emb_val, model_emb_val)  # float('inf')
        mutual_path = os.path.join(path, "mutualinfo.pt")
        torch.save(mutualinfo.state_dict(), mutual_path)
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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, indexs) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = torch.mean(criterion(pred, true))

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        torch.cuda.empty_cache()
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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(get_gpu_memory())
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, indexs) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                with higher.innerloop_ctx(self.model, model_optim) as (
                    fmodel,
                    diffopt,
                ):
                    # build the inner connection
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = fmodel(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = fmodel(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )

                            f_dim = -1 if self.args.features == "MS" else 0
                            outputs = outputs[:, -self.args.pred_len :, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                                self.device
                            )
                            loss_value = torch.mean(
                                criterion(outputs, batch_y), dim=(1, 2)
                            ).reshape(-1, 1)
                    else:
                        if self.args.output_attention:
                            outputs = fmodel(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = fmodel(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss_value = torch.mean(
                            criterion(outputs, batch_y), dim=(1, 2)
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
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss_value = torch.mean(
                            criterion(outputs, batch_y), dim=(1, 2)
                        ).reshape(-1, 1)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss_value = torch.mean(
                        criterion(outputs, batch_y), dim=(1, 2)
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

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
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
        batch_x, batch_y, batch_x_mark, batch_y_mark, indexs = batch_data
        batch_x = batch_x.float().to(self.device)

        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        dec_inp = (
            torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
            .float()
            .to(self.device)
        )

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = fmodel(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = fmodel(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                loss_value = torch.mean(criterion(outputs, batch_y).reshape(-1, 1))
        else:
            if self.args.output_attention:
                outputs = fmodel(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = fmodel(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == "MS" else 0
            outputs = outputs[:, -self.args.pred_len :, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
            loss_value = torch.mean(criterion(outputs, batch_y).reshape(-1, 1))
        return loss_value

    def test(self, setting, test=0):
        if self.args.model == "TimesNet2":
            self.compute_cka()
        test_data, test_loader = self._get_data(flag="test")

        preds = []
        trues = []
        folder_path = "test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, indexs) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]

                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(
                        shape
                    )
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(
                        shape
                    )

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(
                            shape
                        )
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = "results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return
