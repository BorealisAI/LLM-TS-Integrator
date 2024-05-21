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

import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import subprocess

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def format_scientific_no_leading_zero(x):
    formatted = format(x, ".1e")
    if "e" in formatted:
        number, exponent = formatted.split("e")
        exponent = exponent.replace("-0", "-").replace("+0", "+")
        formatted = number + "e" + exponent
    return formatted


def compute_lags(x, top_k=5):
    x = np.atleast_2d(x)
    q_fft = np.fft.rfft(x, axis=-1)
    res = q_fft * np.conj(q_fft)
    corr = np.fft.irfft(res, axis=-1)
    corr_half = corr.flatten()[1:]
    top_K_indices = np.argsort(corr_half)[-top_k:]
    top_k_lags = top_K_indices + 1
    return top_k_lags


def dataset2description(dataset="Handwriting", task_type=None):
    print("dataset name", dataset)
    # Anomaly detection
    if dataset == "SMD":
        dataset_description = (
            "The SMD is a 5-week-long dataset collected from a large Internet company."
        )
        dimension2feature = []
        dimension_wise = True
    elif dataset == "MSL":
        dataset_description = "The MSL dataset includes time series telemetry data from the Mars Science Laboratory rover."
        dimension2feature = []
        dimension_wise = True
    elif dataset == "SMAP":
        dataset_description = "The SMAP dataset includes time series telemetry data from the Soil Moisture Active Passive satellite."
        dimension2feature = []
        dimension_wise = True
    elif dataset == "SWaT":
        dataset_description = (
            "SWaT is obtained from 51 sensors of the critical infrastructure system."
        )
        dimension2feature = []
        dimension_wise = True
    elif dataset == "PSM":
        dataset_description = "PSM (Pooled Server Metrics) is collected internally from multiple application server nodes at eBay."
        dimension2feature = []
        dimension_wise = True

    # classfication
    if dataset == "Handwriting":
        dataset_description = "The Handwriting dataset contains 3D accelerometer data from smartwatch motion while writing the alphabet."
        dimension2feature = [
            "horizontal wrist acceleration",
            "vertical wrist acceleration",
            "depth wrist acceleration",
        ]
        dimension_wise = False
    elif dataset == "JapaneseVowels":
        dataset_description = "The JapaneseVowels dataset consists of time series data from nine male speakers' utterances of Japanese vowels. "
        dimension2feature = []
        dimension_wise = False
    elif dataset == "EthanolConcentration":
        dataset_description = "The EthanolConcentration dataset captures spectral data from whisky bottles for ethanol concentration classification. "
        dimension2feature = ["wavelength", "wavelength", "wavelength"]
        dimension_wise = True
    elif dataset == "FaceDetection":
        dataset_description = "The FaceDetection dataset represents MEG recordings across 10 subjects, each with 580-590 trials of face/scramble. "
        dimension2feature = []
        dimension_wise = True
    elif dataset == "Heartbeat":
        dataset_description = "The Heartbeat dataset compares normal and abnormal heart sounds from diverse subjects. "
        dimension2feature = []
        dimension_wise = True
    elif dataset == "PEMS-SF":
        dataset_description = "The PEMS-SF dataset comprises 963 sensors' data, sampled in 144 ten-minute intervals for weekday classification. "
        dimension2feature = []
        dimension_wise = True
    elif dataset == "SelfRegulationSCP1":
        dataset_description = "The dataset contains EEG data with 6 dimensions, sampled at 896 instances per trial for cursor movement via brain signals. "
        dimension2feature = []
        dimension_wise = True
    elif dataset == "SelfRegulationSCP2":
        dataset_description = "The dataset involves ALS patient's brain signals controlling a cursor for potential classification. "
        dimension2feature = []
        dimension_wise = True
    elif dataset == "SpokenArabicDigits":
        dataset_description = "The dataset comprises spoken Arabic digits, represented by 13 Mel Frequency Cepstral Coefficients. "
        dimension2feature = []
        dimension_wise = False
    elif dataset == "UWaveGestureLibrary":
        dataset_description = "The dataset features three-axis accelerometer data for eight simple gestures. "
        dimension2feature = []
        dimension_wise = False

    # imputation
    if dataset in ["ETTh1", "ETTm1", "ETTh2", "ETTm2"]:
        dataset_description = "The Electricity Transformer Temperature is a crucial indicator in electric power long-term deployment."
        dimension2feature = [
            "high useful load",
            "high useless load",
            "middle useful load",
            "middle useless load",
            "low useful load",
            "low useless load",
            "oil temperature",
        ]
        dimension_wise = False
    elif dataset == "electricity":
        dataset_description = "The dataset comprises time-series data on electricity consumption with measurements every 15 minutes."
        dimension2feature = []
        dimension_wise = True
    elif dataset == "weather":
        dataset_description = "The Weather dataset is a 21-dimensional time series collected every 10 minutes ."
        dimension2feature = []
        dimension_wise = True

    # short-term forecasting
    if dataset in ["Yearly"]:
        dataset_description = (
            "The M4-Yearly dataset contains yearly collected univariate data."
        )
        dimension2feature = []
        dimension_wise = False
    elif dataset in ["Quarterly"]:
        dataset_description = (
            "The M4-Quarterly dataset contains quarterly collected univariate data."
        )
        dimension2feature = []
        dimension_wise = False
    elif dataset in ["Monthly"]:
        dataset_description = (
            "The M4-Monthly dataset contains monthly collected univariate data."
        )
        dimension2feature = []
        dimension_wise = False
    elif dataset in ["Weekly"]:
        dataset_description = (
            "The M4-Weekly dataset contains weekly collected univariate data."
        )
        dimension2feature = []
        dimension_wise = False
    elif dataset in ["Daily"]:
        dataset_description = (
            "The M4-Weekly dataset contains daily collected univariate data."
        )
        dimension2feature = []
        dimension_wise = False
    elif dataset in ["Hourly"]:
        dataset_description = (
            "The M4-Hourly dataset contains hourly collected univariate data."
        )
        dimension2feature = []
        dimension_wise = False

    # long-term forecasting
    if dataset == "traffic":
        dataset_description = "The Traffic dataset provides hourly transportation data across California's freeway system."
        dimension2feature = []
        dimension_wise = True
    elif dataset == "exchange_rate":
        dataset_description = "The Exchange dataset collects daily exchange rates of eight foreign countries."
        dimension2feature = [
            "Australia",
            "British",
            "Canada",
            "Switzerland",
            "China",
            "Japan",
            "New Zealand",
            "Singapore",
        ]
        dimension_wise = False
    if dataset in ["illness"]:
        dataset_description = "The Influenza-like Illness dataset tracks weekly flu-like symptom prevalence in the U.S."
        dimension2feature = []
        dimension_wise = False

    if task_type:
        dataset_description = "The task is " + task_type + "." + dataset_description

    return dataset_description, dimension2feature, dimension_wise


def process_sample_raw(sample, dataset_description, dimension2feature, dimension_wise):
    text = []
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1))
    if dimension_wise:
        for row_index in range(sample.shape[1]):
            row = sample[:, row_index]
            row_content = [format_scientific_no_leading_zero(x) for x in row]
            sample_formatted = ", ".join(row_content)
            tmp = f"{sample_formatted}."
            text.append(tmp)
    else:
        text = ""
        for row_index in range(sample.shape[1]):
            row = sample[:, row_index]
            row_content = [format_scientific_no_leading_zero(x) for x in row]
            sample_formatted = ", ".join(row_content)
            tmp = f"{sample_formatted}."
            text = text + tmp
    return text


def process_sample(sample, dataset_description, dimension2feature, dimension_wise):
    text = []
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1))
    if dimension_wise:
        for row_index in range(sample.shape[1]):
            row = sample[:, row_index]
            row_content = [format_scientific_no_leading_zero(x) for x in row]
            sample_formatted = ", ".join(row_content)
            if len(dimension2feature):
                tmp = (
                    f"{dataset_description} "
                    f"The {row_index} dimension represents {dimension2feature[row_index]}. "
                    f"The content: {sample_formatted}. "
                    f"Input statistics: "
                    f"min {format_scientific_no_leading_zero(min(row))}, "
                    f"max {format_scientific_no_leading_zero(max(row))}, "
                    f"median {format_scientific_no_leading_zero(np.median(row))}, "
                    f"top 5 lags {compute_lags(row)}."
                )
            else:
                tmp = (
                    f"{dataset_description} "
                    f"The {row_index} dimension content: {sample_formatted}. "
                    f"Input statistics: "
                    f"min {format_scientific_no_leading_zero(min(row))}, "
                    f"max {format_scientific_no_leading_zero(max(row))}, "
                    f"median {format_scientific_no_leading_zero(np.median(row))}, "
                    f"top 5 lags {compute_lags(row)}."
                )
            text.append(tmp)
    else:
        text = f"{dataset_description} "
        for row_index in range(sample.shape[1]):
            row = sample[:, row_index]
            row_content = [format_scientific_no_leading_zero(x) for x in row]
            sample_formatted = ", ".join(row_content)
            if len(dimension2feature):
                tmp = (
                    f"The {row_index} dimension represents {dimension2feature[row_index]}. "
                    f"The content: {sample_formatted}. "
                    f"Input statistics: "
                    f"min {format_scientific_no_leading_zero(min(row))}, "
                    f"max {format_scientific_no_leading_zero(max(row))}, "
                    f"median {format_scientific_no_leading_zero(np.median(row))}, "
                    f"top 5 lags {compute_lags(row)}."
                )
            else:
                tmp = (
                    f"The {row_index} dimension content: {sample_formatted}. "
                    f"Input statistics: "
                    f"min {format_scientific_no_leading_zero(min(row))}, "
                    f"max {format_scientific_no_leading_zero(max(row))}, "
                    f"median {format_scientific_no_leading_zero(np.median(row))}, "
                    f"top 5 lags {compute_lags(row)}."
                )
            text = text + tmp
    return text


def get_next_batch(vali_loader):
    while True:
        for batch in vali_loader:
            yield batch


def dataset2dimension(dataset="Handwriting"):
    if dataset == "Handwriting":
        return 152
    elif dataset == "JapaneseVowels":
        return 29
    elif dataset == "EthanolConcentration":
        return 1751
    elif dataset == "FaceDetection":
        return 62
    elif dataset == "Heartbeat":
        return 405
    elif dataset == "PEMS-SF":
        return 144
    elif dataset == "SelfRegulationSCP1":
        return 896
    elif dataset == "SelfRegulationSCP2":
        return 1152
    elif dataset == "SpokenArabicDigits":
        return 93
    elif dataset == "UWaveGestureLibrary":
        return 315


def get_gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv"],
            text=True,
            capture_output=True,
            check=True,
        )
        output = result.stdout
        lines = output.strip().split("\n")
        headers = lines[0].split(",")
        values = lines[1].split(",")
        total_memory = values[0].strip()
        used_memory = values[1].strip()
        return f"Total GPU Memory: {total_memory}, Used GPU Memory: {used_memory}"
    except subprocess.CalledProcessError:
        return "Failed to execute nvidia-smi"
