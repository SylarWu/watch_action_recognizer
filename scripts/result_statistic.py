import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        precision[i] = confusion[i][i] / np.sum(confusion[i, :])
        recall[i] = confusion[i][i] / np.sum(confusion[:, i])

    for i in range(n_classes):
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    return precision, recall, f1


def draw_bar(preprocess_strategy, metric, x, y):
    dpi = 120
    plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    plt.title("%s-%s" % (preprocess_strategy, metric))
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.ylim(0, 1)

    for a, b in zip(x, y):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=12)
    plt.xticks(size=10, rotation=-45)
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

    models = set()
    strategies = set()

    for dir_path in os.listdir(checkpoint_path):
        model_name, preprocess_strategy = dir_path.split('-')
        models.add(model_name)
        strategies.add(preprocess_strategy)

    confusions = {}
    accuracys = {}
    precisions = {}
    recalls = {}
    f1s = {}

    for model_name in models:
        for preprocess_strategy in strategies:
            confusion = pd.read_csv(os.path.join(checkpoint_path, '%s-%s' % (model_name, preprocess_strategy),
                                                 'confusion_matrix.csv'), sep=',', header=None)
            confusion = confusion.to_numpy()
            accuracy = calc_accuracy(confusion, confusion.shape[0])
            precision, recall, f1 = calc_precision_recall_f1(confusion, confusion.shape[0])

            confusions['%s-%s' % (model_name, preprocess_strategy)] = confusion
            accuracys['%s-%s' % (model_name, preprocess_strategy)] = accuracy
            precisions['%s-%s' % (model_name, preprocess_strategy)] = precision
            recalls['%s-%s' % (model_name, preprocess_strategy)] = recall
            f1s['%s-%s' % (model_name, preprocess_strategy)] = f1

            print(('%s-%s' % (model_name, preprocess_strategy)).center(100, '='))
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
    shuffle_count = 0
    shuffle_accuracy = [0.0 for model_name in x]
    shuffle_precision = [0.0 for model_name in x]
    shuffle_recall = [0.0 for model_name in x]
    shuffle_f1 = [0.0 for model_name in x]
    for preprocess_strategy in strategies:
        if not preprocess_strategy.startswith('shuffle'):
            y = [np.mean(accuracys["%s-%s" % (model_name, preprocess_strategy)]) for model_name in x]
            draw_bar(preprocess_strategy, 'accuracy', x, y)
            y = [np.mean(precisions["%s-%s" % (model_name, preprocess_strategy)]) for model_name in x]
            draw_bar(preprocess_strategy, 'precision', x, y)
            y = [np.mean(recalls["%s-%s" % (model_name, preprocess_strategy)]) for model_name in x]
            draw_bar(preprocess_strategy, 'recall', x, y)
            y = [np.mean(f1s["%s-%s" % (model_name, preprocess_strategy)]) for model_name in x]
            draw_bar(preprocess_strategy, 'f1', x, y)
        else:
            shuffle_count += 1
            for i in range(len(x)):
                shuffle_accuracy[i] += np.mean(accuracys["%s-%s" % (x[i], preprocess_strategy)])
                shuffle_precision[i] += np.mean(precisions["%s-%s" % (x[i], preprocess_strategy)])
                shuffle_recall[i] += np.mean(recalls["%s-%s" % (x[i], preprocess_strategy)])
                shuffle_f1[i] += np.mean(f1s["%s-%s" % (x[i], preprocess_strategy)])
    shuffle_accuracy = [_ / shuffle_count for _ in shuffle_accuracy]
    shuffle_precision = [_ / shuffle_count for _ in shuffle_precision]
    shuffle_recall = [_ / shuffle_count for _ in shuffle_recall]
    shuffle_f1 = [_ / shuffle_count for _ in shuffle_f1]
    draw_bar('shuffle', 'accuracy', x, shuffle_accuracy)
    draw_bar('shuffle', 'precision', x, shuffle_precision)
    draw_bar('shuffle', 'recall', x, shuffle_recall)
    draw_bar('shuffle', 'f1', x, shuffle_f1)
