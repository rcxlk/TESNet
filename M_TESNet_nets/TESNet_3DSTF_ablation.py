import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv, get_activation
from .CPCABlock import CPCA
from torchvision.models.shufflenetv2 import channel_shuffle
# from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        P5 = self.lateral_conv0(feat3)
        # -------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        # -------------------------------------------#
        P5_upsample = self.upsample(P5)
        # -------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        # -------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        P4 = self.reduce_conv1(P5_upsample)
        # -------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        # -------------------------------------------#
        P4_upsample = self.upsample(P4)
        # -------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        # -------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        P3_out = self.C3_p3(P4_upsample)
        return P3_out


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[16, 32, 64], act="silu"):
        super().__init__()
        Conv = BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------#
            x = self.stems[k](x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class CAF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        hidden = in_channel // 2

        # 跨通道注意力用于当前帧特征
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, hidden, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, in_channel, 1, 1),
            nn.Sigmoid()
        )

        # 空间注意力用于参考帧特征
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # 双特征交互门控（控制信息来源占比）
        self.gate_conv = nn.Sequential(
            BaseConv(in_channel * 2, in_channel, 3, 1, act="sigmoid"),
        )

        # 融合卷积
        self.fuse_conv = nn.Sequential(
            BaseConv(in_channel, hidden, 3, 1),
            BaseConv(hidden, out_channel, 1, 1)
        )

    def forward(self, r_feat, c_feat):
        # 通道注意力强化当前帧
        c_attn = self.channel_attn(c_feat)
        c_feat_attn = c_feat * c_attn

        # 空间注意力强化参考帧
        max_pool = torch.max(r_feat, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(r_feat, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        r_attn = self.spatial_attn(sa_input)
        r_feat_attn = r_feat * r_attn

        # 拼接并计算融合门控
        cat_feat = torch.cat([c_feat_attn, r_feat_attn], dim=1)
        gate = self.gate_conv(cat_feat)
        fused = gate * c_feat_attn + (1 - gate) * r_feat_attn

        # 融合输出
        out = self.fuse_conv(fused)
        return out


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channel_1, in_channel_2, out_channel, kernel_size=1, stride=1, padding=0, groups=8):
        super(FeatureFusionBlock, self).__init__()

        self.groups = groups

        self.spatio_attn = CPCA(in_channel_1)
        self.temporal_attn = CPCA(in_channel_2)
        self.fusion = nn.Conv2d(in_channel_1+in_channel_2, out_channel, kernel_size, stride, padding)

    def forward(self, spatio_feature, temporal_feature):
        spatio_feature = self.spatio_attn(spatio_feature)
        temporal_feature = self.temporal_attn(temporal_feature)

        cat_feature = channel_shuffle(torch.cat((spatio_feature, temporal_feature), dim=1),
                                      groups=self.groups)
        fusion_feature = self.fusion(cat_feature)
        return fusion_feature


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


class STLSTMNet(nn.Module):
    def __init__(self, num_layer=2, num_hidden=16, in_channel=1, out_channel=16, length=5,
                 kernel_size=3, stride=1):
        super(STLSTMNet, self).__init__()
        self.skip_init = True

        self.num_layer = num_layer
        self.num_hidden = num_hidden
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_frame = length
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        cell_list = []
        for i in range(self.num_layer):
            cell_in_channel = self.in_channel if i == 0 else self.num_hidden
            cell_list.append(SpatioTemporalLSTMCell(cell_in_channel, self.num_hidden,
                                                    self.kernel_size, self.stride))
        self.cell_list = nn.ModuleList(cell_list)

        conv_in_list = []
        for t in range(self.num_frame):
            if t == 0:
                conv_in_channel = self.in_channel
            else:
                conv_in_channel = self.num_hidden + self.in_channel
            conv_in_list.append(BaseConv(conv_in_channel, self.in_channel, self.kernel_size, self.stride))
        self.conv_in_list = nn.ModuleList(conv_in_list)

        conv_out_list = []
        for t in range(self.num_frame):
            conv_out_list.append(BaseConv(self.num_hidden, self.out_channel, self.kernel_size, self.stride))
        self.conv_out_list = nn.ModuleList(conv_out_list)

        self.conv_m = BaseConv(self.num_hidden, self.out_channel, self.kernel_size, self.stride)

    def forward(self, frames_tensor, alpha=0.5):
        # batch length channel height width
        frames = frames_tensor.contiguous()
        batch, length, _, height, width = frames.size()
        device = frames.device

        h_t = []
        c_t = []

        for i in range(self.num_layer):
            zeros = torch.zeros([batch, self.num_hidden, height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden, height, width]).to(device)

        out_frames = []
        for t in range(length):
            if t == 0:
                conv_input = frames[:, t, :, :, :]
            else:
                # input_frame = self.conv_in(torch.cat((h_t[-1], frames[:, t, :, :, :]), dim=1))
                conv_input = torch.cat((h_t[-1], frames[:, t, :, :, :]), dim=1)

            input_frame = self.conv_in_list[t](conv_input)
            h_t[0], c_t[0], memory, _, _ = self.cell_list[0](input_frame, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layer):
                h_t[i], c_t[i], memory, _, _ = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            out_frames.append(self.conv_out_list[t](h_t[-1]))
        out = self.conv_m(memory)

        return out, out_frames


# 多尺度空间注意力模块
class MultiScaleSpaceAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleSpaceAttention, self).__init__()
        self.conv1 = BaseConv(in_channels, in_channels, 3, 1)
        self.conv2 = BaseConv(in_channels, in_channels, 5, 1)
        self.conv3 = BaseConv(in_channels, in_channels, 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算多尺度卷积
        attn_map1 = self.conv1(x)
        attn_map2 = self.conv2(x)
        attn_map3 = self.conv3(x)

        # 融合多尺度特征
        attention_map = (attn_map1 + attn_map2 + attn_map3) / 3
        attention_map = self.sigmoid(attention_map)

        return x * attention_map  # 按空间位置加权


class CrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Sequential(BaseConv(channels, channels // 2, 1, 1),)
        self.key = nn.Sequential(BaseConv(channels, channels // 2, 1, 1),)
        self.value = nn.Sequential(BaseConv(channels, channels, 1, 1),)
        self.scale = (channels // 2) ** -0.5
        self.attn_conv = BaseConv(channels // 2, channels, 1, 1, act='sigmoid')

        # 深度可分卷积
        self.conv_final = BaseConv(channels, channels, 3, 1)

    def forward(self, current_feat, context_feat):
        q = self.query(current_feat)
        k = self.key(context_feat)
        v = self.value(context_feat)

        attn = q * k * self.scale
        attn = self.attn_conv(attn)
        weighted_value = attn * v

        enhanced_feat = self.conv_final(weighted_value)

        return current_feat + enhanced_feat  # 融合当前特征和增强特征


class Neck(nn.Module):
    def __init__(self, num_layer=2, num_hidden=16, channels=[128,256,512], num_frame=5,
                 use_space=True, use_stlstm=True, use_fusion=True):
        super().__init__()

        self.use_stlstm = use_stlstm  # 是否启用时序分支
        self.use_space = use_space  # 是否启用空间分支
        self.use_fusion = use_fusion  # 是否启用融合模块

        self.num_frame = num_frame

        # 空间-关键帧与参考帧融合
        if self.use_space:
            self.conv_text = nn.Sequential(BaseConv(channels[0] * (self.num_frame - 1), channels[0] * 2, 3, 1),
                                          BaseConv(channels[0]*2,channels[0],3,1))
            self.cross_attn = CrossAttention(channels[0])
            self.space_attn = MultiScaleSpaceAttention(channels[0])


        # 时间-提取时空特征
        if self.use_stlstm:
            self.stlstm = STLSTMNet(num_layer, num_hidden, channels[0], channels[0], self.num_frame, 3, 1)
            self.st_attn = nn.ModuleList()
            for i in range(self.num_frame):
                self.st_attn.append(FeatureFusionBlock(channels[0], channels[0], channels[0], 1, 1, 0))
        self.conv_sth_mix = nn.Sequential(BaseConv(channels[0] * self.num_frame, channels[0] * 2, 3, 1),
                                          BaseConv(channels[0]*2,channels[0],3,1))

        # 融合上述两种特征
        if self.use_fusion:
            self.conv_final = CAF(channels[0], channels[0])
        else:
            self.conv_final = BaseConv(channels[0] * 2, channels[0], 1, 1)

    def forward(self, feats):
        f_feats = []

        # 空间
        if self.use_space:
            t_feat = torch.cat([feats[j] for j in range(self.num_frame - 1)],dim=1)
            t_feat = self.conv_text(t_feat)  # 提取多帧上下文信息
            w_feat = self.cross_attn(feats[-1], t_feat)  # 交叉注意力融合多帧上下文信息和当前帧，得到增强的当前帧
            tmp_feat = feats[-1] + w_feat   # 将增强后的当前帧与原始当前帧相加
            tmp_feat = self.space_attn(tmp_feat)
        else:
            # 关闭空间分支，直接用最后一帧特征当空间特征
            tmp_feat = feats[-1]

        # 时间
        if self.use_stlstm:
            st_feats = torch.stack(feats, dim=1)
            st_feat, p_feats = self.stlstm(st_feats)  # 将多个帧送入stlstm进行提取时空特征及单帧特征
            fu_feats = []
            for i in range(self.num_frame):
                fu_feats.append(self.st_attn[i](p_feats[i], st_feat))
            stm_feat = self.conv_sth_mix(torch.cat(fu_feats, dim=1))
        else:
            # 关闭时序分支，简单对多帧特征cat并conv
            stm_feat = self.conv_sth_mix(torch.cat(feats, dim=1))

        # 融合模块
        if self.use_fusion:
            c_feat = self.conv_final(stm_feat, tmp_feat)  # 将上述两种特征进行加权融合
        else:
            # 关闭融合模块，简单融合
            c_feat = self.conv_final(torch.cat((stm_feat, tmp_feat), dim=1))
        f_feats.append(c_feat)

        return f_feats


class BaseConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=(3, 3, 3), stride=(1, 1, 1), groups=1, bias=False, act="silu"):
        super().__init__()

        padding = tuple((k - 1) // 2 for k in ksize)  # T H W
        self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Backbone3D(nn.Module):
    def __init__(self, num_frames=5, in_channel=3):
        super().__init__()
        self.p1_down = nn.Sequential(BaseConv3D(in_channel, 16, (num_frames, 1, 1), (1, 1, 1)),
                                     BaseConv3D(16, 32, (1, 3, 3), (1, 2, 2)),)
        self.p2_down = nn.Sequential(BaseConv3D(32, 32, (num_frames, 1, 1), (1, 1, 1)),
                                     BaseConv3D(32, 64, (1, 3, 3), (1, 2, 2)),)
        self.p3_down = nn.Sequential(BaseConv3D(64, 64, (num_frames, 1, 1), (1, 1, 1)),
                                     BaseConv3D(64, 128, (1, 3, 3), (1, 2, 2)),)

        self.spatial_conv = nn.Sequential(BaseConv(128, 128, 3, 1),
                                          BaseConv(128, 128, 3, 1), )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x1 = self.p1_down(x)
        x2 = self.p2_down(x1)
        x3 = self.p3_down(x2)
        feat = []
        for i in range(T):
            feat.append(self.spatial_conv(x3[:, :, i]))
        return feat


class TESNet(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=5,
                 use_3d=True, use_space=True, use_stlstm=True, use_fusion=True):
        super(TESNet, self).__init__()

        self.use_3d = use_3d
        self.use_space = use_space
        self.use_stlstm = use_stlstm
        self.use_fusion = use_fusion

        self.num_frame = num_frame

        if self.use_3d:
            self.backbone = Backbone3D(num_frames=num_frame, in_channel=3)
        else:
            self.backbone = YOLOPAFPN(0.33, 0.50)

        # -----------------------------------------#
        #   尺度感知模块
        # -----------------------------------------#
        self.neck = Neck(channels=[128], num_frame=num_frame,
                         use_space=use_space, use_stlstm=use_stlstm, use_fusion=use_fusion)
        # ----------------------------------------------------------#
        #   head
        # ----------------------------------------------------------#
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[128], act="silu")

    def forward(self, inputs):
        if self.use_3d:
            feat = self.backbone(inputs)
        else:
            feat = []
            for i in range(self.num_frame):
                feat.append(self.backbone(inputs[:, :, i, :, :]))
        """[b,128,32,32][b,256,16,16][b,512,8,8]"""

        if self.neck:
            feat = self.neck(feat)
        outputs = self.head(feat)
        return outputs  # 计算损失那边 的anchor 应该是 [1, M, 4] size的


if __name__ == "__main__":

    from yolo_training import YOLOLoss

    net = TESNet(num_classes=1, num_frame=5)

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    bs = 2
    a = torch.randn(bs, 3, 5, 512, 512)
    # torch.onnx.export(net, a, "DSTCFNet_3D.onnx")

    out = net(a)
    for item in out:
        print(item.size())

    # start_time = time.time()
    # for _ in range(100):  # 测试 100 次推理时间
    #     _ = model(inputs)
    # end_time = time.time()
    # total_time = end_time - start_time
    # fps = 100 / total_time

    flops, params = profile(net, (a,))

    print("-" * 50)
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + ' G')
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + ' M')
    # print(f"FPS: {fps:.2f}")

    # yolo_loss = YOLOLoss(num_classes=1, fp16=False, strides=[16])
    #
    # target = torch.randn([bs, 1, 5]).cuda()
    # target = nn.Softmax()(target)
    # target = [item for item in target]
    #
    # loss = yolo_loss(out, target)
    # print(loss)


