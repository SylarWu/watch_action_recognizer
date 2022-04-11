import os
import json
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

model_colors = {
    "mixer": '#FFD700',
    "vit": '#87CEEB',
    "resnet": '#483D8B',
    "lstm": '#F786E0',
}


def load_json(file_path: os.path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return data


model_params = load_json("./params.json")
model_flops = load_json("./flops.json")


class ModelStatistic:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metric = {
            'normal': {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
            },
            'user': {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            },
            'shuffle': {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            }
        }
        self.params = model_params[model_name]
        self.flops = model_flops[model_name]

    def add_metric(self, strategy, metric_name, metric_value):
        self.metric[strategy][metric_name].append(metric_value)

    def info(self):
        info = "{"
        info += "model_name:%s," % self.model_name
        info += "params(m):%.4f," % self.params
        info += "flops(m):%.4f," % self.flops
        for strategy_name in ["normal", "user", "shuffle"]:
            for metric_name in ["accuracy", "precision", "recall", "f1"]:
                info += "%s-%s:%.4f," % (strategy_name, metric_name, np.mean(self.metric[strategy_name][metric_name]))
        info += "}"
        return info

def to_csv(model_statistics):
    with open("statistics.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for model_statistic in model_statistics:
            row = [model_statistic.model_name, model_statistic.params, model_statistic.flops]
            for strategy_name in ["normal", "user", "shuffle"]:
                for metric_name in ["accuracy", "precision", "recall", "f1"]:
                    row.append(np.mean(model_statistic.metric[strategy_name][metric_name]))
            writer.writerow(row)


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


def draw_model_metric_bar(method, strategy, model_name, metric_name, metric_value):
    print("%s-%s-%s-%s" % (method, strategy, model_name, metric_name))
    plt.clf()
    dpi = 120
    plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    plt.title("%s-%s-%s" % (method, strategy, model_name))
    color = "#20afdf"
    if strategy == 'normal':
        plt.xlabel("The (2*i - 2*i+1)th Attempts")
    elif strategy == 'user':
        plt.xlabel("Participant ID")

    plt.ylabel(metric_name)

    mean_value = np.mean(metric_value)

    x = list(range(len(metric_value)))
    plt.bar(x, metric_value, color=color)
    plt.bar(len(metric_value), mean_value, color="#ccaaee")

    for a, b in zip(x, metric_value):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=14)
    plt.text(len(metric_value), mean_value, '%.2f' % mean_value, ha='center', va='bottom', fontsize=14)

    x_label = list(range(len(metric_value) + 1))
    x_label[len(metric_value)] = "mean"
    plt.xticks(list(range(len(metric_value) + 1)), x_label)
    plt.savefig(os.path.join('./result_pics', "%s-%s-%s-%s" % (method, strategy, model_name, metric_name)))
    # plt.show()


def draw_global_flops_metric_scatter(method, strategy, metric_name, model_statistics):
    plt.clf()
    dpi = 120
    plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    plt.title("%s-%s" % (method, strategy))
    plt.xlabel('FLOPS(M)')
    plt.ylabel(metric_name)
    plt.xscale("log")
    plt.grid(True)

    temp_params = []
    temp_metrics = []
    for model_name, model_statistic in model_statistics[method].items():
        if model_name.startswith('resnet'):
            temp_params.append(model_statistic.flops)
            temp_metrics.append(np.mean(model_statistic.metric[strategy][metric_name]))
    plt.scatter(temp_params, temp_metrics, marker='.', c=model_colors['resnet'], s=80, label='resnet')

    temp_params = []
    temp_metrics = []
    for model_name, model_statistic in model_statistics[method].items():
        if model_name.startswith('vit'):
            temp_params.append(model_statistic.flops)
            temp_metrics.append(np.mean(model_statistic.metric[strategy][metric_name]))
    plt.scatter(temp_params, temp_metrics, marker=',', c=model_colors['vit'], s=80, label='vit')

    temp_params = []
    temp_metrics = []
    for model_name, model_statistic in model_statistics[method].items():
        if model_name.startswith('mixer'):
            temp_params.append(model_statistic.flops)
            temp_metrics.append(np.mean(model_statistic.metric[strategy][metric_name]))
    plt.scatter(temp_params, temp_metrics, marker='v', c=model_colors['mixer'], s=80, label='mixer')

    temp_params = []
    temp_metrics = []
    for model_name, model_statistic in model_statistics[method].items():
        if model_name.startswith('lstm'):
            temp_params.append(model_statistic.flops)
            temp_metrics.append(np.mean(model_statistic.metric[strategy][metric_name]))
    plt.scatter(temp_params, temp_metrics, marker='*', c=model_colors['lstm'], s=80, label='lstm')

    plt.legend()
    plt.show()


def draw_ablation(method, strategies, metric_name, special_model, model_statistics):
    plt.clf()
    dpi = 120
    plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    plt.title("%s" % (special_model))
    scales = ["es", "ms", "s"]
    patches = [8, 16, 32]
    colores = ['#FFD700', '#87CEEB', '#483D8B']
    plt.ylabel(metric_name)
    for index, strategy in enumerate(strategies):

        x = np.arange(len(scales))
        bar_width = 0.2
        plt.subplot(1, len(strategies), index + 1)
        if index == 0:
            plt.ylabel(metric_name)
        plt.title("%s" % strategy)
        plt.xticks(x + bar_width, scales)

        for patch_index, patch in enumerate(patches):
            temp_metrics = []
            for scale in scales:
                temp_metrics.append(np.mean(
                    model_statistics[method]["%s_%s_%d" % (special_model, scale, patch)].metric[strategy][metric_name]))
            plt.ylim(int(min(temp_metrics) * 100) // 10 * 10 / 100,
                     int(max(temp_metrics) * 100) // 10 * 10 / 100 + 0.08)
            plt.bar(x + (patch_index) * bar_width, temp_metrics, width=bar_width, color=colores[patch_index])
    plt.legend(["patch size=%d" % _ for _ in patches])
    plt.show()


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

    methods = sorted(methods)
    strategies = sorted(strategies)
    seq_lens = sorted(seq_lens)
    models = sorted(models)

    model_statistics = {
        'upsampling': {
            model_name: ModelStatistic(model_name) for model_name in models
        },
        'padding': {
            model_name: ModelStatistic(model_name) for model_name in models
        },
    }

    for method in methods:
        for strategy in strategies:
            for seq_len in seq_lens:
                for model_name in models:
                    if not os.path.exists(os.path.join(checkpoint_path, '%s-%s-%d-%s' %
                                                                        (method,
                                                                         strategy,
                                                                         seq_len,
                                                                         model_name,),
                                                       'confusion_matrix.csv')):
                        continue
                    confusion = pd.read_csv(os.path.join(checkpoint_path, '%s-%s-%d-%s' %
                                                         (method,
                                                          strategy,
                                                          seq_len,
                                                          model_name,),
                                                         'confusion_matrix.csv'), sep=',', header=None)
                    confusion = confusion.to_numpy()
                    accuracy = calc_accuracy(confusion, confusion.shape[0])
                    precision, recall, f1 = calc_precision_recall_f1(confusion, confusion.shape[0])

                    stra, _ = strategy.split("_")
                    model_statistics[method][model_name].add_metric(stra, "accuracy", accuracy)
                    model_statistics[method][model_name].add_metric(stra, "precision", np.mean(precision))
                    model_statistics[method][model_name].add_metric(stra, "recall", np.mean(recall))
                    model_statistics[method][model_name].add_metric(stra, "f1", np.mean(f1))

    method = "padding"
    for model_name, statistic in model_statistics[method].items():
        print(statistic.info())

    for model_name, statistic in model_statistics[method].items():
        for strategy_name, metrics in statistic.metric.items():
            for metric_name, metric_value in metrics.items():
                draw_model_metric_bar("padding", strategy_name, model_name, metric_name, metric_value)

    for strategy_name in ["normal", "user", "shuffle"]:
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            draw_global_flops_metric_scatter(method, strategy_name, metric_name, model_statistics)

    special_model = 'vit'
    special_metric = 'accuracy'
    draw_ablation(method, ['normal', 'user', 'shuffle'], special_metric, special_model, model_statistics)

    to_csv(model_statistics[method].values())


