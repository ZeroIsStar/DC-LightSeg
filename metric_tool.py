import numpy as np
import torch

class SegEvaluator:
    def __init__(self, class_num=4):
        if class_num == 1:
            class_num = 2
        self.num_class = class_num
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        # --- 新增：边界统计量 ---
        self.TPb = 0
        self.FPb = 0
        self.FNb = 0

    # ---------- 原有方法保持不变 ----------
    def kappa(self, OA):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = np.sum(self.confusion_matrix)
        pe = np.dot(pe_rows, pe_cols) / (sum_total ** 2 + 1e-8)
        return (OA - pe) / (1 - pe + 1e-8)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    # ---------- 新增：边界处理 ----------
    def _bw_boundary(self, bw):
        """3×3 膨胀 - 原图 = 边界（numpy 版）"""
        h, w = bw.shape
        padded = np.pad(bw, 1, mode='constant')
        # 8-邻域
        neigh = (padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
                 padded[1:-1, :-2] |                    padded[1:-1, 2:] |
                 padded[2:, :-2]   | padded[2:, 1:-1] | padded[2:, 2:])
        return neigh & (~bw)   # 只保留新变 1 的像素

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # 常规 confusion matrix
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        # 边界指标
        gt_b   = self._bw_boundary(gt_image.astype(bool))
        pred_b = self._bw_boundary(pre_image.astype(bool))
        self.TPb += np.logical_and(gt_b, pred_b).sum()
        self.FPb += np.logical_and(~gt_b, pred_b).sum()
        self.FNb += np.logical_and(gt_b, ~pred_b).sum()

    def reset(self):
        self.confusion_matrix[:] = 0
        self.TPb = self.FPb = self.FNb = 0

    # ---------- 最终指标 ----------
    def matrix(self, class_index):
        """不再带 gt_image/pre_image 参数"""
        metric = {}
        # 常规指标
        diag = np.diag(self.confusion_matrix)
        sum_row = self.confusion_matrix.sum(axis=1)
        sum_col = self.confusion_matrix.sum(axis=0)
        precision_cls = diag / (sum_row + 1e-8)
        recall_cls    = diag / (sum_col + 1e-8)
        OA = diag.sum() / (self.confusion_matrix.sum() + 1e-8)
        iou_cls = diag / (sum_row + sum_col - diag + 1e-8)

        metric['0_IoU'] = iou_cls[0]
        metric['1_IoU'] = iou_cls[1]
        metric['IoU']   = np.nanmean(iou_cls)
        metric['Precision'] = precision_cls[class_index]
        metric['Recall']    = recall_cls[class_index]
        metric['OA']  = OA
        metric['F1']  = (2 * precision_cls[class_index] * recall_cls[class_index]) / \
                        (precision_cls[class_index] + recall_cls[class_index] + 1e-8)
        metric['Kappa'] = self.kappa(OA)

        # 边界指标
        eps = 1e-8
        prec_b = self.TPb / (self.TPb + self.FPb + eps)
        rec_b  = self.TPb / (self.TPb + self.FNb + eps)
        bf1 = 2 * prec_b * rec_b / (prec_b + rec_b + eps)
        biou = self.TPb / (self.TPb + self.FPb + self.FNb + eps)
        metric['BF1']  = bf1
        metric['BIoU'] = biou
        return metric


# ================== 随机数据演示 ==================
if __name__ == '__main__':
    evaluator = SegEvaluator(1)          # 二分类
    pred = torch.randint(0, 2, (1, 1, 128, 128)).numpy()
    label = torch.randint(0, 2, (1, 1, 128, 128)).numpy()

    evaluator.reset()
    for p, l in zip(pred, label):
        evaluator.add_batch(l.squeeze(), p.squeeze())

    met = evaluator.matrix(1)   # 只关心类别 1
    print('F1  :', met['F1'])
    print('BF1 :', met['BF1'])
    print('BIoU:', met['BIoU'])
