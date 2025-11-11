import json
import os
import colorsys
# from S_AGPCNet_nets.network import AGPCNet as AGPCnet
# from S_ACMNet_nets.segmentation import ASKCResNetFPN as ACMnet_FPN
# from S_MSHNet_nets.MSHNet import MSHNet as MSHnet
# from S_ISTDUNet_nets.ctNet import ISTDU_Net as ISTDUnet
# # from S_SCTransNet_nets.SCTransNet import SCTransNet as SCTransnet
# from S_SANet_nets.SA import SABody as SAnet
# from M_STMENet_nets.STME import STNetwork as STMEnet
# from M_DTUM_nets.model_alcnet import ALCNet_DTUM as DTUMnet
# from S_DNANet_nets.model_DNANet import DNANet as DNANet
# from M_DSTCFNet_nets.DSTCFNet_No3DConv import DSTCFNet as DSTCFnet_BSTF
# from M_DSTCFNet_nets.DSTCFNet_BS import DSTCFNet as DSTCFnet_BS
# from M_DSTCFNet_nets.DSTCFNet_BST import DSTCFNet as DSTCFnet_BST
# from M_DSTCFNet_nets.DSTCFNet_3DSTF1 import DSTCFNet as DSTCFnet_3DSTF
# from S_ResUNet_nets.model_ResUNet import ResUNet as ResUnet
from M_DSTCFNet_nets.DSTCFNet_3DSTF_ablation import DSTCFNet as DSTCFnet_3DSTF_ablation
from M_SSTNet_nets.Network import Network as SSTnet
from M_DMIST_nets.LASNet import LASNet as LASnet
# from M_DSTCFNet_nets.DSTCFNet_3DSTF_single import DSTCFNet as DSTCFnet_3DSTF_single
# from M_TMPNet_nets.TASA_modified import Tasanet as TMPnet
from M_DTUM_nets.model_res_UNet import res_UNet_DTUM as DTUMnet_Res
from S_DNANet_nets.model_DNANet import DNANet as DNAnet

from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression

#---------------------------------------------------------------------------#
#   map_mode用于指定该文件运行时计算的内容
#   map_mode为0代表整个map计算流程，包括获得预测结果、计算map。
#   map_mode为1代表仅仅获得预测结果。
#   map_mode为2代表仅仅获得计算map。
#---------------------------------------------------------------------------#
map_mode            = 0
#-------------------------------------------------------#
#   指向了验证集标签与图片路径
#-------------------------------------------------------#
cocoGt_path         = r'D:\rcx-code\DetectionCode\SIMDataSet_allSimData\sim_dataset\annotation\val_annotations.json'
dataset_img_path    = r'D:\rcx-code\DetectionCode\SIMDataSet_allSimData\sim_dataset\images\val_sequences'
# cocoGt_path         = r'E:\code\rcx_code\target_detection\DAUB\annotations\val.json'
# dataset_img_path    = r'E:\code\rcx_code\target_detection\DAUB\images\test'
#-------------------------------------------------------#
#   结果输出的文件夹，默认为map_out
#-------------------------------------------------------#
temp_save_path      = 'map_out/coco_eval'
#-------------------------------------------------------#
#   设置帧数
#-------------------------------------------------------#
num_frame = 5

class MAP_vid(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_dat a下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path": r'logs_DSTCFNet_BSTF/loss_2025_06_06_16_05_37/last_epoch_weights.pth',
        # "model_path": r'logs_DTUMNet_ALC/loss_2025_06_10_11_11_11/last_epoch_weights.pth',
        "model_path": r'logs_DNANet_real_sim/loss_2025_09_14_11_33_51/last_epoch_weights.pth',
        # "model_path": r'logs_DSTCFNet_BS/loss_2025_06_10_22_36_41/last_epoch_weights.pth',
        # "model_path": r'logs_DSTCFNet_BST/loss_2025_06_11_08_52_53/last_epoch_weights.pth',
        # "model_path": r'logs_SANet/loss_2025_06_05_08_10_40/best_epoch_weights.pth',
        # "model_path": r'logs_STMENet/loss_2025_06_08_07_30_07/last_epoch_weights.pth',
        # "model_path": r'logs_ISTDUNet_real_sim/loss_2025_09_14_06_40_38/last_epoch_weights.pth',
        # "model_path": r'logs_ACMNet_FPN/loss_2025_06_12_05_49_27/last_epoch_weights.pth',
        # "model_path": r'logs_MSHNet/loss_2025_06_13_08_55_26/last_epoch_weights.pth',
        # "model_path": r'logs_DSTCFNet_t1/loss_2025_06_15_00_24_14/last_epoch_weights.pth',
        # "model_path": r'logs_ResUNet/loss_2025_06_13_15_31_32/last_epoch_weights.pth',

        # "model_path": r'logs_SSTNet_real_sim/loss_2025_09_13_09_35_13/last_epoch_weights.pth',
        # "model_path": r'logs_TESNet_real_sim/loss_2025_09_12_17_14_07/last_epoch_weights.pth',
        # "model_path": r'logs_TMPNet_real_sim/loss_2025_09_13_19_18_03/last_epoch_weights.pth',
        # "model_path": r'logs_DTUMNet_Res_real_sim1/loss_2025_09_13_20_32_37/ep080-loss2.355-val_loss3.487.pth',
        # "model_path": r'logs_LASNet_real_sim/loss_2025_09_13_19_47_05/ep060-loss4.038-val_loss5.949.pth',

        "classes_path"      : r'model_data/classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [512, 512],
        #---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.1,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        # self.net    = Tasanet(self.num_classes, num_frame=num_frame)
        # self.net = AGPCnet(num_classes=1, backbone='resnet18')
        # self.net = SCTransnet(n_channels=3, n_classes=1)
        # self.net = DSTCFnet_BS(num_classes=1, num_frame=num_frame)
        # self.net = DSTCFnet_BST(num_classes=1, num_frame=num_frame)
        # self.net = DTUMnet(num_classes=1, num_frames=num_frame)
        # self.net = DNANet(num_classes=1, input_channels=3)
        # self.net = SAnet(num_classes=1)
        # self.net = STMEnet(num_classes=1, num_frame=num_frame)
        # self.net = ISTDUnet(num_classes=1, in_channels=3)
        self.net = DNAnet(num_classes=1, input_channels=3)
        # self.net = ACMnet_FPN(num_classes=1)
        # self.net = MSHnet(num_classes=1, input_channels=3)
        # self.net = DSTCFnet_3DSTF(num_classes=1, num_frame=num_frame)
        # self.net = ResUnet(num_classes=1, input_channels=3)

        # self.net = DSTCFnet_3DSTF_ablation(num_classes=1, num_frame=num_frame)

        # self.net = DSTCFnet_3DSTF_ablation(num_classes=1, num_frame=num_frame)
        # self.net = SSTnet(num_classes=1, num_frame=num_frame)
        # self.net = TMPnet(num_classes=1, num_frame=num_frame)
        # self.net = DTUMnet_Res(num_classes=1, num_frames=num_frame)
        # self.net = LASnet(num_classes=1, num_frame=num_frame)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
                
     #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_id, images, results):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(images[0])[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        images       = [cvtColor(image) for image in images]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = [resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image) for image in images]
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = [np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)) for image in image_data]
        # (3, 640, 640) -> (3, 16, 640, 640)
        image_data = np.stack(image_data, axis=1)
        
        image_data  = np.expand_dims(image_data, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            outputs = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if outputs[0] is None: 
                return results

            top_label   = np.array(outputs[0][:, 6], dtype = 'int32')
            top_conf    = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes   = outputs[0][:, :4]

        for i, c in enumerate(top_label):
            result                      = {}
            top, left, bottom, right    = top_boxes[i]

            result["image_id"]      = int(image_id)
            result["category_id"]   = clsid2catid[c]
            result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"]         = float(top_conf[i])
            results.append(result)
        return results

def get_history_imgs(line, num=num_frame):
    line = os.path.normpath(line).replace('\\', '/')
    dir_path = line.replace(line.split('/')[-1],'')
    file_type = line.split('.')[-1]
    index = int(line.split('/')[-1][:-4])
    
    return [os.path.join(dir_path,  "%d.%s" % (max(id, 0),file_type)) for id in range(index-num+1, index + 1)]


if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    cocoGt      = COCO(cocoGt_path)
    ids         = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    if map_mode == 0 or map_mode == 1:
        yolo = MAP_vid(confidence = 0.001, nms_iou = 0.65)

        with open(os.path.join(temp_save_path, 'eval_results.json'),"w") as f:
            results = []
            for image_id in tqdm(ids):
                image_path  = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
                images = get_history_imgs(image_path)
                images = [Image.open(item) for item in images]
                # image       = Image.open(image_path)
                results     = yolo.detect_image(image_id, images, results)
            json.dump(results, f)

    if map_mode == 0 or map_mode == 2:
        cocoDt      = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox')
        # cocoEval.params.iouThrs = np.array([0.5])  # 只评估 mAP@0.5
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        """
        T:iouThrs [0.5:0.05:0.95] T=10 IoU thresholds for evaluation
        R:recThrs [0:0.01:100] R=101 recall thresholds for evaluation
        K: category ids 
        A: [all, small, meduim, large] A=4 
        M: maxDets [1, 10, 100] M=3 max detections per image
        """
        precisions = cocoEval.eval['precision']
        precision_50 = precisions[0,:,0,0,-1]  # 第三为类别 (T,R,K,A,M)
        recalls = cocoEval.eval['recall']
        recall_50 = recalls[0,0,0,-1] # 第二为类别 (T,K,A,M)
        pre = np.mean(precision_50[:int(recall_50*100)])
        F1 = 2*pre*recall_50/(pre+recall_50)
        map50 = np.mean(precision_50[precision_50 > -1])
        print("\n Precision: %.4f, Recall: %.4f, F1: %.4f, mAP50: %.4f" %(pre, recall_50, F1, map50))
        print(" Get map done.")
        
        # 画图
        import matplotlib.pyplot as plt
        plt.figure(1) 
        plt.title('PR Curve')# give plot a title
        plt.xlabel('Recall')# make axis labels
        plt.ylabel('Precision')
        
        x_axis = plt.xlim(0,100)
        y_axis = plt.ylim(0,1.05)
        plt.figure(1)
        plt.plot(precision_50)
        plt.savefig('p-r.png')
        plt.show()

        # ===========================
        # 额外添加 TP, FP, FN 指标统计
        # ===========================
        # T = 0.5  # IoU threshold
        # cat_id = 0  # 类别id (默认第一个类别，可根据 cocoGt.getCatIds() 循环)
        # areaRngIdx = 0  # all
        # maxDetIdx = -1  # maxDet = 100
        # iou_thr_idx = np.where(np.isclose(cocoEval.params.iouThrs, T))[0][0]
        # evalImgs = cocoEval.evalImgs
        # num_imgs = len(cocoGt.getImgIds())
        # TP = 0
        # FP = 0
        # FN = 0
        # for img_idx in range(num_imgs):
        #     eval_data = evalImgs[img_idx * len(cocoEval.params.catIds) + cat_id]
        #     if eval_data is None:
        #         continue
        #     dtMatches = eval_data['dtMatches'][iou_thr_idx]
        #     dtIgnore = eval_data['dtIgnore'][iou_thr_idx]
        #     gtIgnore = eval_data['gtIgnore']
        #     # TP：成功匹配的检测框，且不在忽略列表中
        #     TP += np.sum((dtMatches > 0) & (~dtIgnore))
        #     # FP：未匹配的检测框，且不在忽略列表中
        #     FP += np.sum((dtMatches == 0) & (~dtIgnore))
        #     # FN：未被检测到的GT（不在忽略列表中）
        #     FN += np.sum(~np.array(gtIgnore)) - np.sum((dtMatches > 0) & (~dtIgnore))
        # detection_rate = TP / (TP + FN + 1e-6)
        # false_alarm_rate = FP / (TP + FP + 1e-6)
        # print("\n--- Detection Quality Metrics @IoU=0.5 ---")
        # print(f"TP: {TP}, FP: {FP}, FN: {FN}")
        # print(f"Detection Rate:     {detection_rate:.4f}")
        # print(f"False Alarm Rate:   {false_alarm_rate:.4f}")
        # 检测率 ≈ recall
        detection_rate = recall_50
        # 虚警率
        # 获取评估的每个图像的 TP, FP, FN 信息
        tp = 0
        fp = 0
        fn = 0
        for eval_res in cocoEval.evalImgs:
            if eval_res is None:
                continue
            dtMatches = eval_res.get('dtMatches')  # [T, D]
            gtMatches = eval_res.get('gtMatches')  # [T, G]
            gtIgnore = eval_res.get('gtIgnore')  # [G]
            dtIgnore = eval_res.get('dtIgnore')  # [T, D]

            if dtMatches is None or gtMatches is None:
                continue

            # 使用 IoU = 0.5（T=0）这一个阈值下的匹配信息
            dtMatches = dtMatches[0]
            gtMatches = gtMatches[0]
            dtIgnore = dtIgnore[0]

            tp_img = np.sum((dtMatches > 0) & (dtIgnore == 0))  # 正确检测（不被忽略的匹配框）
            fp_img = np.sum((dtMatches == 0) & (dtIgnore == 0))  # 虚警（无匹配的预测框）
            fn_img = np.sum((gtMatches == 0) & (gtIgnore == 0))  # 漏检（无匹配的GT框）

            tp += tp_img
            fp += fp_img
            fn += fn_img

        false_alarm_rate = fp / (tp + fn) if (tp + fn) > 0 else 0
        # false_alarm_rate = 1 - pre  # ≈ 1 - precision
        print("\n Detection Rate: %.4f, False Alarm Rate: %.4f" % (detection_rate, false_alarm_rate))

        # with open("our_result_2.txt", 'w') as f:
        #     for pred in precision_50:
        #         f.writelines(str(pred)+'\t')

