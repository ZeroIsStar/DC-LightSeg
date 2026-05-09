import os
import torch
import logging
import numpy as np
from Data_loader import DataLoader, cfg
from configs import Loader
from PIL import Image
from metric_tool import SegEvaluator



class evaluator:
    def __init__(self, test_loader = None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), model_weight_dir=None,model_type =None, dataset_name=None):
        self.cfg = cfg
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.loder = Loader(self.model_type)
        self.Test_loader = test_loader
        self.model = self.loder.model.to(device)
        self.model_weight = model_weight_dir
        self.SegEvaluator = SegEvaluator(1)

    def make_dir(self):
        result = 'tf-logs'
        os.makedirs(result, exist_ok=True)
        dataset_dir = os.path.join(result, self.dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        model_dir = os.path.join(dataset_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        loss_function_dir = os.path.join(model_dir, self.cfg.train.loss_function)
        os.makedirs(loss_function_dir, exist_ok=True)
        out_dir = os.path.join(loss_function_dir, 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        return loss_function_dir, out_dir

    def save_test_evaluate(self, log_dir, F1, recall, precision, iou, BF1, BIoU):
        # 记录日志
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别为INFO或者更高级别
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'test_evaluate.log')),  # 将日志写入指定的文件
            ]
        )
        logging.info(
            f'Test_evaluate \n'
            f'F1={F1},\n'
            f'recall={recall},'
            f'precision={precision}\n'
            f'iou={iou},\n'
            f'BF1={BF1},\n'
            f'Biou={BIoU},\n'
        )

    def vis_seg_test_result(self):
        if self.model_weight is None:
            raise ValueError('输入模型权重地址')
        self.model.load_state_dict(torch.load(self.model_weight))   # 加载模型权重
        _, log_dir = self.make_dir()
        # 创建文件夹存储可视化结果
        label_out = os.path.join(log_dir, "label")
        predict_out = os.path.join(log_dir, "predict")
        Diff_out = os.path.join(log_dir, "diff_out")
        # 判断是否有文件夹,没有就创建
        os.makedirs(label_out, exist_ok=True)
        os.makedirs(predict_out, exist_ok=True)
        os.makedirs(Diff_out, exist_ok=True)
        self.model.eval()
        # 定义颜色映射表
        color_map = np.array([[0, 0, 0],  # 类别0: 黑色
                              [255, 255, 255]])  # 类别1: 白色
        with torch.no_grad():
            for batch, sample in enumerate(self.Test_loader):
                img, label, name = sample
                label = label.squeeze(0)
                # 使用颜色映射表将单通道数据映射为3通道数据
                color_image = color_map[label]
                image = Image.fromarray(color_image.astype(np.uint8), mode='RGB')
                # 保存图像为 PNG 文件
                image.save(os.path.join(label_out, f'{str(name[0])}.png'),format='PNG')
                self.model.cuda()
                outputs = self.model(img.cuda())
                outputs = torch.argmax(torch.sigmoid(outputs), dim=1).cpu()
                # 将张量转换为 PIL 图像
                outputs = outputs.squeeze(0)
                # 使用颜色映射表将单通道数据映射为3通道数据
                color_image = color_map[outputs]
                # 对比图
                # 定义颜色映射
                diff_map = np.zeros((outputs.shape[0], outputs.shape[1], 3), dtype=np.uint8)  # 初始化颜色映射图
                # TP: 预测为1，标签也为1
                diff_map[(outputs == 1) & (label == 1)] = [0, 255, 0]  # 绿色
                # TN: 预测为0，标签也为0
                diff_map[(outputs == 0) & (label == 0)] = [255, 255, 255]  # 白色
                # FP: 预测为1，但标签为0
                diff_map[(outputs == 1) & (label == 0)] = [0, 0, 255]  # 蓝色
                # FN: 预测为0，但标签为1
                diff_map[(outputs == 0) & (label == 1)] = [255, 0, 0]  # 红色
                diff_map_rgba = np.zeros((outputs.shape[0], outputs.shape[1], 4), dtype=np.uint8)
                diff_map_rgba[:, :, :3] = diff_map  # 将 RGB 数据复制到 RGBA 的前 3 个通道
                diff_map_rgba[:, :, 3] = 255  # 默认为完全不透明
                mask = (diff_map_rgba[:, :, 0] == 255) & (diff_map_rgba[:, :, 1] == 255) & (
                            diff_map_rgba[:, :, 2] == 255)
                diff_map_rgba[mask, 3] = 0  # 将白色部分的 Alpha 设置为 0
                image = Image.fromarray(color_image.astype(np.uint8), mode='RGB')
                diff_image = Image.fromarray(diff_map_rgba, mode='RGBA')
                # 保存图像为 PNG 文件
                image.save(os.path.join(predict_out, f"{name[0]}.png"),format='PNG')
                diff_image.save(os.path.join(Diff_out, f"{name[0]}.png"),format='PNG')
        print('输出完毕')

    def test(self):
        _, log_dir = self.make_dir()
        if self.model_weight is None:
            raise ValueError('输入模型权重地址')
        self.model.load_state_dict(torch.load(self.model_weight))  # 加载模型权重
        self.model.eval()
        self.SegEvaluator.reset()
        with torch.no_grad():
            for batch, sample in enumerate(self.Test_loader):
                img, label, _ = sample
                img, label = img.cuda(), label.cuda()
                outputs = self.model(img)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                output = outputs.cpu().numpy().astype(np.int8)
                label = label.cpu().numpy().astype(np.int8)
                for p, l in zip(output, label):
                    self.SegEvaluator.add_batch(l.squeeze(), p.squeeze())
            metric = self.SegEvaluator.matrix(1)
            Iou = metric['1_IoU']
            recall = metric['Recall']
            precision = metric['Precision']
            F1 = metric['F1']
            BF1 = metric['BF1']
            biou = metric['BIoU']
            # Iou = TP / (TP + FN + FP + 1e-5)
            # recall = TP / (TP + FN + 1e-5)
            # precision = TP / (TP + FP + 1e-5)
            # F1 = 2 * (precision * recall) / (precision + recall)
            # Specificity = TN / (TN + FP + 1e-5)
            self.save_test_evaluate(log_dir, F1=F1, recall=recall, precision=precision, iou=Iou,BF1=BF1,BIoU=biou)


if __name__ == '__main__':
    # model_list = ['Unet']
    # dataset_list = ['palu', 'WenChuan']

    Data_loader = DataLoader(dataset_name='Wenchuan')
    # _, _, Test_loader = Data_loader.get_dataloader()
    _,Val_loader,T = Data_loader.get_dataloader()
    evaluator = evaluator(test_loader= T, model_type='DC_light_bifpn_cat', dataset_name='Wenchuan',
                          model_weight_dir=r'autodl-tmp/Bijie/DC_light_bifpn_cat/Tversky_loss_lovasz_a0.5_b0.5/2026_05_09/00_15_13/BEST_epoch114_iou0.7852_Acc0.9748_recall0.8604_pre0.8998_F1_0.8797.pth')
    evaluator.test()
    evaluator.vis_seg_test_result()
