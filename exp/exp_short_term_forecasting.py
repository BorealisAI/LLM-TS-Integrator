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
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.tools import (
    dataset2description,
    process_sample,
    get_next_batch
)
from utils.cka import linear_CKA, kernel_CKA
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
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

warnings.filterwarnings("ignore")


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)
        self.hidden_size = None
        if os.path.exists(self.args.root_path + "/feature_data.pt"):
            self.llm_emb = (
                torch.load(self.args.root_path + "/feature_data.pt")
                .to(self.device)
                .detach()
            )

    def _build_model(self):
        if self.args.data == "m4":
            self.args.pred_len = M4Meta.horizons_map[
                self.args.seasonal_patterns
            ]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
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

    def _select_criterion(self, loss_name="MSE"):
        if loss_name == "MSE":
            return nn.MSELoss(reduce=False)
        elif loss_name == "MAPE":
            return mape_loss()
        elif loss_name == "MASE":
            return mase_loss()
        elif loss_name == "SMAPE":
            return smape_loss()

    def _mutual_information(self, mutualinfo_model, indexs, prob_mutual):
        train_data, train_loader = self._get_data(flag="train")
        timeseries = train_loader.dataset.timeseries
        model_emb = []

        indexs = torch.LongTensor(indexs).cpu().numpy().tolist()
        limit = self.llm_emb.shape[0] - 1
        indexs = [min(x, limit) for x in indexs]

        llm_emb = self.llm_emb[indexs]
        for i in range(len(indexs)):
            index = indexs[i]
            sample = (
                torch.from_numpy(np.array(timeseries[index]).reshape(1, -1, 1))
                .float()
                .to(self.device)
            )
            sample = sample[:, 0 : self.args.seq_len, :]
            sample_emb = self.model.forecast(
                sample, None, None, None, return_hidden=True
            )
            model_emb.append(sample_emb.mean(1).reshape(1, -1))
        model_emb = torch.cat(model_emb, dim=0).to(self.device)
        mutualinfo_estimation = mutualinfo_model(llm_emb, model_emb, prob_mutual)
        return mutualinfo_estimation

    def pretrain_model(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting + "_prt")
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, indexs) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len :, f_dim:].to(
                    self.device
                )
                loss_value = torch.mean(
                    criterion(
                        batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark
                    )
                )
                loss_sharpness = mse(
                    (outputs[:, 1:, :] - outputs[:, :-1, :]),
                    (batch_y[:, 1:, :] - batch_y[:, :-1, :]),
                )
                loss = loss_value

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
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
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
        timeseries = train_loader.dataset.timeseries
        N = min(len(timeseries), 5000)

        first_layers = []
        last_layers = []
        self.model.eval()
        for index in range(N):
            sample = (
                torch.from_numpy(np.array(timeseries[index]).reshape(1, -1, 1))
                .float()
                .to(self.device)
            )
            sample = sample[:, 0 : self.args.seq_len, :]
            with torch.no_grad():
                first_layer, last_layer = self.model.forecast(
                    sample, None, None, None, return_hidden=False, return_first=True
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
        timeseries = train_loader.dataset.timeseries
        N = min(len(timeseries), 10000)
        model_emb = []
        self.model.eval()
        for index in range(N):
            sample = (
                torch.from_numpy(np.array(timeseries[index]).reshape(1, -1, 1))
                .float()
                .to(self.device)
            )
            sample = sample[:, 0 : self.args.seq_len, :]
            with torch.no_grad():
                sample_emb = self.model.forecast(
                    sample, None, None, None, return_hidden=True
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
        #
        llm_emb = self.llm_emb[0:N]
        val_len = int(N * 0.3)
        llm_emb_val, llm_emb_train = llm_emb[0:val_len, :], llm_emb[val_len:, :]
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
        # begin our model
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
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
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        print("begin training")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, indexs) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
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
                    outputs = fmodel(batch_x, None, dec_inp, None)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                        self.device
                    )

                    batch_y_mark = batch_y_mark[
                        :, -self.args.pred_len :, f_dim:
                    ].to(self.device)
                    loss_value = criterion(
                        batch_x,
                        self.args.frequency_map,
                        outputs,
                        batch_y,
                        batch_y_mark,
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

                outputs = self.model(batch_x, None, dec_inp, None)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len :, f_dim:].to(
                    self.device
                )
                loss_value = criterion(
                    batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark
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
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
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
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        dec_inp = (
            torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
            .float()
            .to(self.device)
        )

        outputs = fmodel(batch_x, None, dec_inp, None)
        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.args.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

        batch_y_mark = batch_y_mark[:, -self.args.pred_len :, f_dim:].to(self.device)
        loss_value = criterion(
            batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark
        ).reshape(-1, 1)
        return torch.mean(loss_value)

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat(
                [x[:, -self.args.label_len :, :], dec_inp], dim=1
            ).float()
            # encoder - decoder
            outputs = torch.zeros(
                (B, self.args.pred_len, C)
            ).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i] : id_list[i + 1], :, :] = (
                    self.model(
                        x[id_list[i] : id_list[i + 1]],
                        None,
                        dec_inp[id_list[i] : id_list[i + 1]],
                        None,
                    )
                    .detach()
                    .cpu()
                )
            f_dim = -1 if self.args.features == "MS" else 0
            outputs = outputs[:, -self.args.pred_len :, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = torch.mean(
                criterion(
                    x.detach().cpu()[:, :, 0],
                    self.args.frequency_map,
                    pred[:, :, 0],
                    true,
                    batch_y_mark,
                )
            )

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag="train")
        _, test_loader = self._get_data(flag="test")
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat(
                [x[:, -self.args.label_len :, :], dec_inp], dim=1
            ).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i] : id_list[i + 1], :, :] = self.model(
                    x[id_list[i] : id_list[i + 1]],
                    None,
                    dec_inp[id_list[i] : id_list[i + 1]],
                    None,
                )

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == "MS" else 0
            outputs = outputs[:, -self.args.pred_len :, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        print("test shape:", preds.shape)

        # result save
        folder_path = "./m4_results/" + self.args.model + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(
            preds[:, :, 0], columns=[f"V{i + 1}" for i in range(self.args.pred_len)]
        )
        forecasts_df.index = test_loader.dataset.ids[: preds.shape[0]]
        forecasts_df.index.name = "id"
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + "_forecast.csv")

        print(self.args.model)
        file_path = "./m4_results/" + self.args.model + "/"
        if (
            "Weekly_forecast.csv" in os.listdir(file_path)
            and "Monthly_forecast.csv" in os.listdir(file_path)
            and "Yearly_forecast.csv" in os.listdir(file_path)
            and "Daily_forecast.csv" in os.listdir(file_path)
            and "Hourly_forecast.csv" in os.listdir(file_path)
            and "Quarterly_forecast.csv" in os.listdir(file_path)
        ):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print("smape:", smape_results)
            print("mape:", mape)
            print("mase:", mase)
            print("owa:", owa_results)
        else:
            print(
                "After all 6 tasks are finished, you can calculate the averaged index"
            )
        return
