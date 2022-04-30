import sklearn.metrics as Eval
import numpy as np


def Confusion_matrix(Refer, Pred, label_sample=None):
    if label_sample is None:
        label_sample = [0, 1, 2]
    Pred = np.int64(Pred).flatten()
    Refer = np.int64(Refer).flatten()
    current = Eval.confusion_matrix(Refer, Pred, labels=label_sample)  # 不考虑空值所在像素
    print(current)
    return current


def Evaluation(M, display=False):
    M = np.int64(M)
    total = np.sum(M)
    intersection = np.diag(M)
    ground_truth_set = M.sum(axis=1)
    predicted_set = M.sum(axis=0)

    union = ground_truth_set + predicted_set - intersection
    union[union == 0] = 1  # 3个都是0
    IoU = intersection / union.astype(np.float32) * 100
    mIoU = np.mean(IoU)

    fre = ground_truth_set / total
    fwIoU = np.sum(fre * IoU)

    po = np.sum(intersection) / total
    OA = po * 100

    pc = np.sum(ground_truth_set * predicted_set) / np.square(total)
    if pc == 1:  # 只有单一种类且完全预测正确
        Kappa = po * 100
    else:
        Kappa = (po - pc) / (1 - pc) * 100

    ground_truth_set[ground_truth_set == 0] = 1  # intersection 一定也是0
    Recall = intersection / ground_truth_set
    AA = np.mean(Recall)* 100

    predicted_set[predicted_set == 0] = 1  # intersection 一定也是0
    Precision = intersection / predicted_set
    R = Precision + Recall
    R[R == 0] = 1  # Precision * Recall 一定也是0
    F1_Score = np.mean(2 * (Precision * Recall) / R) * 100

    if display:
        print('============')
        print('F1-score: {}%'.format(F1_Score))
        print('Noisy-IoU: {}%'.format(IoU[0]))
        print('Clean-IoU:{}%'.format(IoU[1]))
        print('Over-smooth-IoU:{}%'.format(IoU[2]))
        print('mIoU:{}%'.format(mIoU))
        print('fwIoU:{}%'.format(fwIoU))
        print('OA:{}%'.format(OA))
        print('AA:{}%'.format(AA))
        print('Kappa:{}%'.format(Kappa))
    return IoU, mIoU, fwIoU, OA, AA, Kappa, F1_Score