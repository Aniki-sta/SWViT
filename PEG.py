# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
import math
from functools import partial
# from utils.version_utils import digit_version
'''
二维卷积
'''

# class ConditionalPositionEncoding(nn.Module):
#     def __init__(self, embed_dim=768, stride=1, init_cfg=None):
#         super(ConditionalPositionEncoding, self).__init__()
#         self.proj = nn.Conv2d(
#             embed_dim,
#             embed_dim,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=True,
#             groups=embed_dim)
#         self.stride = stride

#     def forward(self, x, hw_shape):
#         B, N, C = x.shape
#         H, W = hw_shape
#         if N > H * W + 1:
#             raise ValueError(f"Sequence length N ({N}) does not match H*W ({H*W})")
#         feat_token = x[:, 1:, :]  # Exclude cls_token
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         if self.stride == 1:
#             x = self.proj(cnn_feat) + cnn_feat
#         else:
#             x = self.proj(cnn_feat)
#         x = x.flatten(2).transpose(1, 2)
#         x = torch.cat((x, x.new_zeros((B, 1, C))), dim=1)  # Re-add cls_token
#         return x


'''
深度可分离卷积加入残差连接(效果最好)
'''
# class ConditionalPositionEncoding(nn.Module):
#     def __init__(self, embed_dim=768, stride=1, init_cfg=None):
#         super(ConditionalPositionEncoding, self).__init__()
#         # Depthwise convolution
#         self.depthwise_conv = nn.Conv2d(
#             embed_dim, 
#             embed_dim, 
#             kernel_size=3, 
#             stride=stride, 
#             padding=1, 
#             groups=embed_dim,  # This is what makes it depthwise
#             bias=False)
#         # Pointwise convolution
#         self.pointwise_conv = nn.Conv2d(
#             embed_dim, 
#             embed_dim, 
#             kernel_size=1, 
#             bias=True)

#         self.stride = stride

#     def forward(self, x, hw_shape):
#         B, N, C = x.shape
#         H, W = hw_shape
#         if N > H * W + 1:
#             raise ValueError(f"Sequence length N ({N}) does not match H*W ({H*W})")
#         feat_token = x[:, 1:, :]  # Exclude cls_token
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        
#         # Apply depthwise convolution
#         depthwise_output = self.depthwise_conv(cnn_feat)
#         # Apply pointwise convolution
#         pointwise_output = self.pointwise_conv(depthwise_output)
        
#         # Apply residual connection
#         if self.stride == 1:
#             residual_output = cnn_feat + pointwise_output
#         else:
#             # When stride is not 1, we need to adjust the size of cnn_feat to match pointwise_output
#             cnn_feat = nn.functional.interpolate(cnn_feat, size=(H, W), mode='nearest')
#             residual_output = cnn_feat + pointwise_output
        
#         # Reshape and concatenate cls_token
#         residual_output = residual_output.flatten(2).transpose(1, 2)
#         x = torch.cat((x[:, :1, :], residual_output), dim=1)  # Re-add cls_token
#         return x
'''
SRU卷积

'''
from ScConv import ScConv, SRU, CRU

class ConditionalPositionEncoding(nn.Module):
    def __init__(self, embed_dim=768, stride=1, init_cfg=None):
        super(ConditionalPositionEncoding, self).__init__()
        # Depthwise convolution
        self.ScConv = ScConv(
            op_channel=embed_dim, 
            gate_treshold = 0.5,
            alpha = 1/2,
            squeeze_radio = 2 ,
            group_size = 2,
            group_kernel_size = 3,
            )
        # self.SRU = SRU(
        #     oup_channels = embed_dim,
        #     group_num = 16,
        #     gate_treshold = 0.5,
        #     torch_gn = True
        # )
        # self.CRU = CRU(
        #     op_channel= embed_dim,             
        #     alpha = 1/2,
        #     squeeze_radio = 2 ,
        #     group_size = 2,
        #     group_kernel_size = 3,           
        # )
        # self.stride = stride

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        if N > H * W + 1:
            raise ValueError(f"Sequence length N ({N}) does not match H*W ({H*W})")
        feat_token = x[:, 1:, :]  # Exclude cls_token
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        
        # ScConv 卷积
        # ScConv_output = self.ScConv(cnn_feat)
        # 先SRU再CRU
        # CRU_output = self.CRU(cnn_feat)
        SRU_output = self.ScConv(cnn_feat)

        cnn_feat = nn.functional.interpolate(cnn_feat, size=(H, W), mode='nearest')
        residual_output = cnn_feat + SRU_output
        
        # Reshape and concatenate cls_token
        residual_output = residual_output.flatten(2).transpose(1, 2)
        x = torch.cat((x[:, :1, :], residual_output), dim=1)  # Re-add cls_token
        return x
    

'''
小波卷积
'''

# from wtconv import WTConv2d
# class ConditionalPositionEncoding(nn.Module):
#     def __init__(self, embed_dim=768, stride=1, init_cfg=None):
#         super(ConditionalPositionEncoding, self).__init__()
#         # Depthwise convolution
#         self.wtconv = WTConv2d(
#             in_channels = embed_dim,
#             out_channels = embed_dim,
#             kernel_size=5, 
#             stride=1, 
#             bias=True, 
#             wt_levels=1, 
#             wt_type='db1'
#             )
#         # self.stride = stride

#     def forward(self, x, hw_shape):
#         B, N, C = x.shape
#         H, W = hw_shape
#         if N > H * W + 1:
#             raise ValueError(f"Sequence length N ({N}) does not match H*W ({H*W})")
#         feat_token = x[:, 1:, :]  # Exclude cls_token
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        
#         # Apply depthwise convolution
#         wtconv_output = self.wtconv(cnn_feat)
        
#         # Apply residual connection
#         # if self.stride == 1:
#         #     residual_output = cnn_feat
#         # else:
#         #     # When stride is not 1, we need to adjust the size of cnn_feat to match pointwise_output
#         cnn_feat = nn.functional.interpolate(cnn_feat, size=(H, W), mode='nearest')
#         residual_output = cnn_feat + wtconv_output
        
#         # Reshape and concatenate cls_token
#         residual_output = residual_output.flatten(2).transpose(1, 2)
#         x = torch.cat((x[:, :1, :], residual_output), dim=1)  # Re-add cls_token
#         return x