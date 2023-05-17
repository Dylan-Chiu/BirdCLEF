import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt


def get_confusion_matrix(pred, truth):
    # 计算混淆矩阵
    confusion_mat = confusion_matrix(y_true=truth, y_pred=pred)
    return confusion_mat


def show_heatmap(confusion_mat):
    total_samples = confusion_mat.sum(axis=1, keepdims=True)
    confusion_mat_percentage = confusion_mat / total_samples

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(confusion_mat_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig(f'temp.png', dpi=500, bbox_inches='tight') # 解决图片不清晰，不完整的问题
    plt.show()


def get_precision_recall_f1(pred, truth):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=truth, y_pred=pred, average='macro')
    return precision, recall, f1

# 测试
if __name__ == '__main__':
    truth = [2, 0, 2, 2, 0, 1]
    pred = [0, 0, 2, 2, 0, 2]
    confusion_matrix = get_confusion_matrix(pred, truth)
    print(confusion_matrix)
    # show_heatmap(confusion_matrix)
    precision, recall, f1 = get_precision_recall_f1(pred, truth)
    print(precision, recall, f1)
