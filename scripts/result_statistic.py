import os
import sys

import torch
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_model_summary import summary

model_colors = {
    "mixer": '#FFD700',
    "vit": '#87CEEB',
    "resnet": '#483D8B',
    "lstm": '#F786E0',
}


class StrategyStatistic:
    def __init__(self, models):
        self.count = 0
        self.accuracy = {}
        self.precision = {}
        self.recall = {}
        self.f1 = {}
        for model_name in models:
            self.accuracy[model_name] = 0
            self.precision[model_name] = 0
            self.recall[model_name] = 0
            self.f1[model_name] = 0

    def average(self):
        self._average_metric(self.accuracy)
        self._average_metric(self.precision)
        self._average_metric(self.recall)
        self._average_metric(self.f1)

    def _average_metric(self, dic: dict):
        for model_name, value in dic.items():
            dic[model_name] = value / self.count


def calc_accuracy(confusion, n_classes):
    correct = 0
    for i in range(n_classes):
        correct += confusion[i][i]
    return correct / np.sum(confusion)


def calc_precision_recall_f1(confusion, n_classes):
    precision = [0 for _ in range(n_classes)]
    recall = [0 for _ in range(n_classes)]
    f1 = [0 for _ in range(n_classes)]

    for i in range(n_classes):
        precision[i] = confusion[i][i] / np.sum(confusion[i, :]) if np.sum(confusion[i, :]) != 0 else 0
        recall[i] = confusion[i][i] / np.sum(confusion[:, i]) if np.sum(confusion[:, i]) != 0 else 0

    for i in range(n_classes):
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

    return precision, recall, f1


def draw_bar(preprocess_strategy, metric, dic: dict):
    dpi = 120
    plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    plt.title("%s-%s" % (preprocess_strategy, metric))
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.ylim(0, 1)

    aftersort = sorted(dic.items(), key=lambda item: item[1])

    for a, b in aftersort:  # 柱子上的数字显示
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=6)
    plt.xticks(size=8, rotation=-45)
    for model_name, strategy in aftersort:
        if model_name.startswith('mixer'):
            plt.bar(model_name, strategy, width=0.4, color=model_colors['mixer'])
        elif model_name.startswith('vit'):
            plt.bar(model_name, strategy, width=0.4, color=model_colors['vit'])
        elif model_name.startswith('resnet'):
            plt.bar(model_name, strategy, width=0.4, color=model_colors['resnet'])
        elif model_name.startswith('lstm'):
            plt.bar(model_name, strategy, width=0.4, color=model_colors['lstm'])
    plt.savefig(os.path.join('./result_pics', "%s-%s" % (preprocess_strategy, metric)))
    plt.show()


def get_backbone(model_name: str):
    from model import resnet, ResNetConfig
    from model import vit, TransformerConfig
    from model import mlp_mixer, MLPMixerConfig
    from model import lstm, LSTMConfig
    if model_name.startswith('resnet'):
        return resnet(model_name, ResNetConfig())
    elif model_name.startswith('vit'):
        return vit(model_name, TransformerConfig())
    elif model_name.startswith('mixer'):
        return mlp_mixer(model_name, MLPMixerConfig())
    elif model_name.startswith('lstm'):
        return lstm(model_name, LSTMConfig())


def get_params(model_name: str):
    backbone = get_backbone(model_name)
    accData, gyrData = torch.randn(1, 3, 224), torch.randn(1, 3, 224)
    info = summary(backbone, accData, gyrData, batch_size=1, max_depth=10)
    _, params = re.search(r"Total params: \S*\n", info).group(0).split(": ")
    params = params.replace(',', '').replace('\n', '')
    return int(params)


if __name__ == '__main__':
    checkpoint_path = os.path.join("/data/wuxilei/watch_action_recognizer/log")
    if not os.path.exists('./result_pics'):
        os.mkdir(os.path.join('./result_pics'))

    methods = set()
    strategies = set()
    seq_lens = set()
    models = set()

    for dir_path in os.listdir(checkpoint_path):
        method, strategy, seq_len, model_name = dir_path.split('-')
        methods.add(method)
        strategies.add(strategy)
        seq_lens.add(int(seq_len))
        models.add(model_name)

    confusions = {}
    accuracys = {}
    precisions = {}
    recalls = {}
    f1s = {}

    for method in methods:
        for strategy in strategies:
            for seq_len in seq_lens:
                for model_name in models:
                    confusion = pd.read_csv(os.path.join(checkpoint_path, '%s-%s-%d-%s' %
                                                         (method,
                                                          strategy,
                                                          seq_len,
                                                          model_name,),
                                                         'confusion_matrix.csv'), sep=',', header=None)
                    confusion = confusion.to_numpy()
                    accuracy = calc_accuracy(confusion, confusion.shape[0])
                    precision, recall, f1 = calc_precision_recall_f1(confusion, confusion.shape[0])

                    confusions['%s-%s-%d-%s' % (method, strategy, seq_len, model_name)] = confusion
                    accuracys['%s-%s-%d-%s' % (method, strategy, seq_len, model_name)] = accuracy
                    precisions['%s-%s-%d-%s' % (method, strategy, seq_len, model_name)] = precision
                    recalls['%s-%s-%d-%s' % (method, strategy, seq_len, model_name)] = recall
                    f1s['%s-%s-%d-%s' % (method, strategy, seq_len, model_name)] = f1

                    print(('%s-%s-%d-%s' % (method, strategy, seq_len, model_name)).center(100, '='))
                    print('Confusion: ')
                    print(confusion)
                    print('Precision: ' + str(precision))
                    print('Recall: ' + str(recall))
                    print('F1: ' + str(f1))
                    print('mAccuracy: ' + str(accuracy))
                    print('mPricision: ' + str(np.mean(precision)))
                    print('mRecall: ' + str(np.mean(recall)))
                    print('mF1: ' + str(np.mean(f1)))

    x = list(models)
    x.sort()

    upsampling_statistics = {
        'normal': StrategyStatistic(x),
        'user': StrategyStatistic(x),
        'shuffle': StrategyStatistic(x),
    }

    for strategy in strategies:
        if strategy.startswith('normal'):
            upsampling_statistics['normal'].count += 1
            for i, model_name in enumerate(x):
                upsampling_statistics['normal'].accuracy[model_name] += \
                    np.mean(accuracys["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['normal'].precision[model_name] += \
                    np.mean(precisions["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['normal'].recall[model_name] += \
                    np.mean(recalls["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['normal'].f1[model_name] += \
                    np.mean(f1s["upsampling-%s-224-%s" % (strategy, model_name)])
        elif strategy.startswith('user'):
            upsampling_statistics['user'].count += 1
            for i, model_name in enumerate(x):
                upsampling_statistics['user'].accuracy[model_name] += \
                    np.mean(accuracys["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['user'].precision[model_name] += \
                    np.mean(precisions["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['user'].recall[model_name] += \
                    np.mean(recalls["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['user'].f1[model_name] += \
                    np.mean(f1s["upsampling-%s-224-%s" % (strategy, model_name)])
        elif strategy.startswith('shuffle'):
            upsampling_statistics['shuffle'].count += 1
            for i, model_name in enumerate(x):
                upsampling_statistics['shuffle'].accuracy[model_name] += \
                    np.mean(accuracys["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['shuffle'].precision[model_name] += \
                    np.mean(precisions["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['shuffle'].recall[model_name] += \
                    np.mean(recalls["upsampling-%s-224-%s" % (strategy, model_name)])
                upsampling_statistics['shuffle'].f1[model_name] += \
                    np.mean(f1s["upsampling-%s-224-%s" % (strategy, model_name)])

    for key, value in upsampling_statistics.items():
        value.average()
        draw_bar('upsampling_' + key, 'accuracy', value.accuracy)
        draw_bar('upsampling_' + key, 'precision', value.precision)
        draw_bar('upsampling_' + key, 'recall', value.recall)
        draw_bar('upsampling_' + key, 'f1', value.f1)

    padding_statistics = {
        'normal': StrategyStatistic(x),
        'user': StrategyStatistic(x),
        'shuffle': StrategyStatistic(x),
    }

    for strategy in strategies:
        if strategy.startswith('normal'):
            padding_statistics['normal'].count += 1
            for i, model_name in enumerate(x):
                padding_statistics['normal'].accuracy[model_name] += \
                    np.mean(accuracys["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['normal'].precision[model_name] += \
                    np.mean(precisions["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['normal'].recall[model_name] += \
                    np.mean(recalls["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['normal'].f1[model_name] += \
                    np.mean(f1s["padding-%s-224-%s" % (strategy, model_name)])
        elif strategy.startswith('user'):
            padding_statistics['user'].count += 1
            for i, model_name in enumerate(x):
                padding_statistics['user'].accuracy[model_name] += \
                    np.mean(accuracys["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['user'].precision[model_name] += \
                    np.mean(precisions["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['user'].recall[model_name] += \
                    np.mean(recalls["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['user'].f1[model_name] += \
                    np.mean(f1s["padding-%s-224-%s" % (strategy, model_name)])
        elif strategy.startswith('shuffle'):
            padding_statistics['shuffle'].count += 1
            for i, model_name in enumerate(x):
                padding_statistics['shuffle'].accuracy[model_name] += \
                    np.mean(accuracys["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['shuffle'].precision[model_name] += \
                    np.mean(precisions["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['shuffle'].recall[model_name] += \
                    np.mean(recalls["padding-%s-224-%s" % (strategy, model_name)])
                padding_statistics['shuffle'].f1[model_name] += \
                    np.mean(f1s["padding-%s-224-%s" % (strategy, model_name)])

    for key, value in padding_statistics.items():
        value.average()
        draw_bar('padding_' + key, 'accuracy', value.accuracy)
        draw_bar('padding_' + key, 'precision', value.precision)
        draw_bar('padding_' + key, 'recall', value.recall)
        draw_bar('padding_' + key, 'f1', value.f1)


    class ModelStatistic:
        def __init__(self):
            self.metric = 0
            self.params = 0

            self.metric_count = 0
            self.params_count = 0

        def add_metric(self, metric):
            self.metric += metric
            self.metric_count += 1

        def add_params(self, params):
            self.params += params
            self.params_count += 1

        def average(self):
            self.metric /= self.metric_count
            self.params /= self.params_count


    model_statistics = {
        'resnet18': ModelStatistic(),
        'resnet34': ModelStatistic(),
        'resnet50': ModelStatistic(),
        'resnet101': ModelStatistic(),
        'lstm_es': ModelStatistic(),
        'lstm_ms': ModelStatistic(),
        'lstm_s': ModelStatistic(),
        'mixer_es': ModelStatistic(),
        'mixer_ms': ModelStatistic(),
        'mixer_s': ModelStatistic(),
        'vit_es': ModelStatistic(),
        'vit_ms': ModelStatistic(),
        'vit_s': ModelStatistic(),
    }

    for key, value in padding_statistics.items():
        for model_name, accuracy in value.accuracy.items():
            params = get_params(model_name)
            model_attributes = model_name.split("_")
            if len(model_attributes) == 1:
                model_statistics[model_attributes[0]].add_params(params)
                model_statistics[model_attributes[0]].add_metric(accuracy)
            else:
                model_statistics[model_attributes[0] + "_" + model_attributes[1]].add_params(params)
                model_statistics[model_attributes[0] + "_" + model_attributes[1]].add_metric(accuracy)

    dpi = 120
    plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    plt.title("%s" % ("accuracy"))
    plt.xlabel('Total Parameters')
    plt.ylabel("accuracy")
    plt.xticks(np.arange(0, 3e7, 3e7 / 20))
    plt.grid(True)
    for model_name, model_statistic in model_statistics.items():
        model_statistic.average()

    temp_params = []
    temp_metrics = []
    for model_name, model_statistic in model_statistics.items():
        if model_name.startswith('resnet'):
            temp_params.append(model_statistic.params)
            temp_metrics.append(model_statistic.metric)
    plt.scatter(temp_params, temp_metrics, marker='.', c=model_colors['resnet'], s=80, label='resnet')

    temp_params = []
    temp_metrics = []
    for model_name, model_statistic in model_statistics.items():
        if model_name.startswith('vit'):
            temp_params.append(model_statistic.params)
            temp_metrics.append(model_statistic.metric)
    plt.scatter(temp_params, temp_metrics, marker=',', c=model_colors['vit'], s=80, label='vit')

    temp_params = []
    temp_metrics = []
    for model_name, model_statistic in model_statistics.items():
        if model_name.startswith('mixer'):
            temp_params.append(model_statistic.params)
            temp_metrics.append(model_statistic.metric)
    plt.scatter(temp_params, temp_metrics, marker='v', c=model_colors['mixer'], s=80, label='mixer')

    temp_params = []
    temp_metrics = []
    for model_name, model_statistic in model_statistics.items():
        if model_name.startswith('lstm'):
            temp_params.append(model_statistic.params)
            temp_metrics.append(model_statistic.metric)
    plt.scatter(temp_params, temp_metrics, marker='*', c=model_colors['lstm'], s=80, label='lstm')

    plt.legend()
    plt.show()
