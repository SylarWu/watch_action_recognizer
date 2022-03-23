import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class StrategyStatistic:
    def __init__(self, n):
        self.count = 0
        self.accuracy = [0.0 for _ in range(n)]
        self.precision = [0.0 for _ in range(n)]
        self.recall = [0.0 for _ in range(n)]
        self.f1 = [0.0 for _ in range(n)]

    def average(self):
        self.accuracy = [_ / self.count for _ in self.accuracy]
        self.precision = [_ / self.count for _ in self.precision]
        self.recall = [_ / self.count for _ in self.recall]
        self.f1 = [_ / self.count for _ in self.f1]


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


def draw_bar(preprocess_strategy, metric, x, y):
    dpi = 120
    plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    plt.title("%s-%s" % (preprocess_strategy, metric))
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.ylim(0, 1)

    for a, b in zip(x, y):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.xticks(size=8, rotation=-45)
    for model_name, strategy in zip(x, y):
        if model_name.startswith('mixer'):
            plt.bar(model_name, strategy, width=0.4, color='#FFD700')
        elif model_name.startswith('vit'):
            plt.bar(model_name, strategy, width=0.4, color='#87CEEB')
        elif model_name.startswith('resnet'):
            plt.bar(model_name, strategy, width=0.4, color='#483D8B')
    plt.savefig(os.path.join('./result_pics', "%s-%s" % (preprocess_strategy, metric)))
    plt.show()


if __name__ == '__main__':
    checkpoint_path = os.path.join("/data/wuxilei/watch_action_recognizer/log")
    if not os.path.exists('./result_pics'):
        os.mkdir(os.path.join('./result_pics'))

    methods = set()
    strategies = set()
    seq_lens = set()
    normalization = set()
    models = set()

    for dir_path in os.listdir(checkpoint_path):
        method, strategy, seq_len, is_normalize, model_name = dir_path.split('-')
        methods.add(method)
        strategies.add(strategy)
        seq_lens.add(int(seq_len))
        normalization.add(is_normalize)
        models.add(model_name)

    confusions = {}
    accuracys = {}
    precisions = {}
    recalls = {}
    f1s = {}

    for method in methods:
        for strategy in strategies:
            for seq_len in seq_lens:
                for is_normalize in normalization:
                    for model_name in models:
                        confusion = pd.read_csv(os.path.join(checkpoint_path, '%s-%s-%d-%s-%s' %
                                                             (method,
                                                              strategy,
                                                              seq_len,
                                                              is_normalize,
                                                              model_name,),
                                                             'confusion_matrix.csv'), sep=',', header=None)
                        confusion = confusion.to_numpy()
                        accuracy = calc_accuracy(confusion, confusion.shape[0])
                        precision, recall, f1 = calc_precision_recall_f1(confusion, confusion.shape[0])

                        confusions['%s-%s-%d-%s-%s' % (method, strategy, seq_len, is_normalize, model_name)] = confusion
                        accuracys['%s-%s-%d-%s-%s' % (method, strategy, seq_len, is_normalize, model_name)] = accuracy
                        precisions['%s-%s-%d-%s-%s' % (method, strategy, seq_len, is_normalize, model_name)] = precision
                        recalls['%s-%s-%d-%s-%s' % (method, strategy, seq_len, is_normalize, model_name)] = recall
                        f1s['%s-%s-%d-%s-%s' % (method, strategy, seq_len, is_normalize, model_name)] = f1

                        print(
                            ('%s-%s-%d-%s-%s' % (method, strategy, seq_len, is_normalize, model_name)).center(100, '='))
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
        'normal': StrategyStatistic(len(x)),
        'user': StrategyStatistic(len(x)),
        'shuffle': StrategyStatistic(len(x)),
    }

    for strategy in strategies:
        if strategy.startswith('normal'):
            upsampling_statistics['normal'].count += 1
            for i, model_name in enumerate(x):
                upsampling_statistics['normal'].accuracy[i] += \
                    np.mean(accuracys["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['normal'].precision[i] += \
                    np.mean(precisions["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['normal'].recall[i] += \
                    np.mean(recalls["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['normal'].f1[i] += \
                    np.mean(f1s["upsampling-%s-224-none-%s" % (strategy, model_name)])
        elif strategy.startswith('user'):
            upsampling_statistics['user'].count += 1
            for i, model_name in enumerate(x):
                upsampling_statistics['user'].accuracy[i] += \
                    np.mean(accuracys["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['user'].precision[i] += \
                    np.mean(precisions["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['user'].recall[i] += \
                    np.mean(recalls["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['user'].f1[i] += \
                    np.mean(f1s["upsampling-%s-224-none-%s" % (strategy, model_name)])
        elif strategy.startswith('shuffle'):
            upsampling_statistics['shuffle'].count += 1
            for i, model_name in enumerate(x):
                upsampling_statistics['shuffle'].accuracy[i] += \
                    np.mean(accuracys["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['shuffle'].precision[i] += \
                    np.mean(precisions["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['shuffle'].recall[i] += \
                    np.mean(recalls["upsampling-%s-224-none-%s" % (strategy, model_name)])
                upsampling_statistics['shuffle'].f1[i] += \
                    np.mean(f1s["upsampling-%s-224-none-%s" % (strategy, model_name)])

    for key, value in upsampling_statistics.items():
        value.average()
        draw_bar('upsampling_' + key, 'accuracy', x, value.accuracy)
        draw_bar('upsampling_' + key, 'precision', x, value.precision)
        draw_bar('upsampling_' + key, 'recall', x, value.recall)
        draw_bar('upsampling_' + key, 'f1', x, value.f1)

    padding_statistics = {
        'normal': StrategyStatistic(len(x)),
        'user': StrategyStatistic(len(x)),
        'shuffle': StrategyStatistic(len(x)),
    }

    for strategy in strategies:
        if strategy.startswith('normal'):
            padding_statistics['normal'].count += 1
            for i, model_name in enumerate(x):
                padding_statistics['normal'].accuracy[i] += \
                    np.mean(accuracys["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['normal'].precision[i] += \
                    np.mean(precisions["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['normal'].recall[i] += \
                    np.mean(recalls["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['normal'].f1[i] += \
                    np.mean(f1s["padding-%s-224-none-%s" % (strategy, model_name)])
        elif strategy.startswith('user'):
            padding_statistics['user'].count += 1
            for i, model_name in enumerate(x):
                padding_statistics['user'].accuracy[i] += \
                    np.mean(accuracys["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['user'].precision[i] += \
                    np.mean(precisions["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['user'].recall[i] += \
                    np.mean(recalls["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['user'].f1[i] += \
                    np.mean(f1s["padding-%s-224-none-%s" % (strategy, model_name)])
        elif strategy.startswith('shuffle'):
            padding_statistics['shuffle'].count += 1
            for i, model_name in enumerate(x):
                padding_statistics['shuffle'].accuracy[i] += \
                    np.mean(accuracys["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['shuffle'].precision[i] += \
                    np.mean(precisions["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['shuffle'].recall[i] += \
                    np.mean(recalls["padding-%s-224-none-%s" % (strategy, model_name)])
                padding_statistics['shuffle'].f1[i] += \
                    np.mean(f1s["padding-%s-224-none-%s" % (strategy, model_name)])

    for key, value in padding_statistics.items():
        value.average()
        draw_bar('padding_' + key, 'accuracy', x, value.accuracy)
        draw_bar('padding_' + key, 'precision', x, value.precision)
        draw_bar('padding_' + key, 'recall', x, value.recall)
        draw_bar('padding_' + key, 'f1', x, value.f1)