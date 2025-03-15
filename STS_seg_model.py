# add
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import copy
import logging
from os.path import join as pjoin
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from . import print_seghead_vit_configs_mine as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from torch.nn.modules.utils import _pair
from scipy import ndimage
# add end
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from .style_adapt_multi_trans_model import Multi_Transformer as Multi_Transformer
#from .test_Fast_multi_trans_model import Multi_Transformer_test as Multi_Transformer

from .style_adapt_eye_atten import Eye_Center
from .adapt_Fast_Class_Activation_Mapping_show import Class_Activation_Mapping, Show_Avg_CAM_Mapping, Show_Pseudo_Cam, Show_Pseudo_Label


import cv2
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
C = 768 # IT has to be same as config.hidden_size
affine_par = True
Cam_Centre = [16,16]
Cam_correction_1 = torch.zeros([1, 6, 32, 32]).cuda()#[2, 5, 32, 32]
Cam_correction_2 = torch.zeros([2, 6, 32, 32]).cuda()#[2, 5, 32, 32]

centre_open = 0 # 1, open the centre space
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
logger = logging.getLogger(__name__)

class VisionTransformer(nn.Module):
    def __init__(self, config, config_multi_trans, img_size=512, num_classes=6, multi_trans_depth = 3, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)

        self.decoder = DecoderCup(config)
        
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.config_multi_trans = config_multi_trans
        self.Pseudo_Cam = Pseudo_Label()
        # aff
        #self.Cam_to_label = cam_to_label()
        self.depth = multi_trans_depth
        self.fuse = Fuse_feature(config,config_multi_trans)

        self.COMPLEX = Complex()

        self.Central_Space = central_space()
    def forward(self, x,content_layers_output,style_layers_output):
        if x.size()[1] == 1:
            x0 = x.repeat(1,3,1,1)
        x0 = x
        complex_x = self.COMPLEX(x)
        x, attn_weights, features, camone_vb, camsix_vb, attn,hidden_get = self.transformer(x0,complex_x)  # x[2, 1024, 768]
        x_all = self.decoder(x, features) #[2, 1024, 768]-->[2, 16, 512, 512]
        logits = self.segmentation_head(x_all)#[2, 16, 512, 512] -- >[2, 6, 512, 512]
        if centre_open == 1:
            centre_features,seg_layer = self.Central_Space(hidden_get,content_layers_output,style_layers_output)
        else:
            centre_features = 0
            seg_layer = 0
        return logits,centre_features,seg_layer

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            #print(weights["Transformer/posembed_input/pos_embedding"].shape)
            #bug
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"][:,:,:,0:self.config.hidden_size], conv=True)) # (16, 16, 3, 768) -- > (16, 16, 3, hidden_size)
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"][0:self.config.hidden_size]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"][0:self.config.hidden_size]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"][0:self.config.hidden_size]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"][:,:,0:self.config.hidden_size])
            

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


class central_space(nn.Module):
    def __init__(self):
        super(central_space, self).__init__()
        self.seg_fuse_conv24 = nn.Conv2d(in_channels=24, out_channels=1, kernel_size=1)
        self.seg_fuse_conv12 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1)
        self.seg_fuse_conv768 = nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1,stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((16, 16))
    def forward(self, seg_layer, content_layers_output, style_layers_output):  # [2, 3, 512, 512]

        seg_layer = torch.stack(seg_layer)

        content_layers_features = [
            layer for layer in content_layers_output
            if layer.shape in [torch.Size([1, 256, 256, 256]), torch.Size([2, 256, 256, 256])]
        ]
        content_layers_features = torch.stack(content_layers_features,dim = 1).squeeze() # [4, 256, 256, 256]

        style_layers_features = [
            layer for layer in style_layers_output
            if layer.shape in [torch.Size([1, 256, 256, 256]), torch.Size([2, 256, 256, 256])]
        ]

        style_layers_features = torch.stack(style_layers_features,dim = 1).squeeze() # [4, 256, 256, 256]
        centre_features = torch.cat([content_layers_features, style_layers_features],0)

        # seg  Preprocessing , out seg_layer is [16, 16]
        seg_layer = seg_layer.permute(3,1,0,2).contiguous() # [12, 2, 1024, 768]--> [768,2, 12, 1024]
        seg_layer = seg_layer.view(768,int(seg_layer.shape[1]*seg_layer.shape[2]) ,32,32)   # [768,2, 12, 1024]--> [768, 24, 1024]
        if seg_layer.shape[1] == 24:
            seg_layer = self.seg_fuse_conv24(seg_layer)
        else:
            seg_layer = self.seg_fuse_conv12(seg_layer)
        seg_layer = seg_layer.permute(1,0,2,3).contiguous()
        seg_layer = self.seg_fuse_conv768(seg_layer).squeeze(dim=1)
        #print(f"New shape444: {seg_layer.shape}")


        # style Preprocessing,  out seg_layer is [16, 16]
        centre_features = self.avg_pool(centre_features)
        centre_features = centre_features.mean(dim=(0, 1), keepdim=True).squeeze(dim=1)
        #print(f"New centre_features: {centre_features.shape}")

        return centre_features,seg_layer


class Complex(nn.Module):
    def __init__(self):
        super(Complex, self).__init__()
        self.complex_maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.complex_avgpool = torch.nn.AvgPool2d(kernel_size=2)


    def forward(self, x):  # [2, 3, 512, 512]
        mean = 0
        var = 0

        for i in range(3):
            imgtr = (x[:,i, ...]).cpu().detach().numpy()
            meantr = np.mean(imgtr)
            vartr = np.std(imgtr)
            mean = mean + abs(meantr)
            var = var + abs(vartr)
            
        #print ("var",var)
        #print ("mean",mean)
        
        if var  > (0.3*mean):  #
            complex_parameter = "move_no_zero"
        else:
            complex_parameter = "constant_no_zero"

        return complex_parameter

class Fuse_feature(nn.Module):
    def __init__(self, config, config_multi_trans):
        super(Fuse_feature, self).__init__()
        C = config_multi_trans.embed_dim # 24
        hidden_size = config.hidden_size
        s = int(8/(config_multi_trans.patch_size))
        self.up = nn.Conv2d(C,hidden_size,kernel_size=3,stride=s,padding = 1,bias=False)
        self.down = nn.Conv2d(int(hidden_size*2),hidden_size,kernel_size=3,stride=1,padding = 1,bias=False)


    def forward(self, x_trans, x_multi): # [2, 1024, 768] & [2, 24, 32, 32] --> [2, 1024, 768]
        x_trans = x_trans.permute(0, 2, 1).contiguous()    # [2, 768, 1024]
        x_trans = x_trans.view(x_trans.shape[0], x_trans.shape[1], int(math.sqrt(x_trans.shape[2])), int(math.sqrt(x_trans.shape[2]))) # [2, 768, 32, 32]
        
        x_multi = 0.1 *x_multi
        x_multi = self.up(x_multi) # [2, 768, 64, 64]

        x_all = torch.cat([x_trans, x_multi],1) 
         #[2, 104, 32, 32]
        x_all = self.down(x_all)
        x_all = x_all.view(x_all.shape[0], x_all.shape[1],int((x_all.shape[2])*(x_all.shape[3])))
        x_all = x_all.permute(0, 2, 1).contiguous()

        return x_all # [2, 1024, 768]

class Pseudo_Label(nn.Module):
    def __init__(self):
        super(Pseudo_Label, self).__init__()
        self.Show_Pseudo_Label = Show_Pseudo_Label()
        self.down = nn.Conv2d(12,6,kernel_size=3,stride=1,padding = 1,bias=False)


    def forward(self, camone_vb, camsix_vb, camone_multi, camsix_multi): # [2, 512, 512], [2, 6, 512, 512], [2, 512, 512], [2, 6, 512, 512]
        '''
        print ("camone_vb", camone_vb.shape)
        print ("camsix_vb", camsix_vb.shape)
        print ("camone_multi", camone_multi.shape)
        print ("camsix_multi", camsix_multi.shape)
        '''
        cams1 = torch.einsum('nhw,nhw->nhw', camone_vb, camone_multi) # [2, 512, 512]

        cams11 = cams1.unsqueeze(dim=1)
        cams2 = torch.einsum('nchw,nchw->nchw', camsix_vb, camsix_multi) # [2, 6, 512, 512]
        cam1to6_vb = camsix_vb
        cam1to6_multi = camsix_multi
        #camone_vb = camone_vb.unsqueeze(dim=1)
        #camone_multi = camone_multi.unsqueeze(dim=1)

        for i in range(6):
            cam1to6_vb[:,i,:,:] = camone_vb
            cam1to6_multi[:,i,:,:] = camone_multi
        cams3 = torch.einsum('nchw,nchw->nchw', cam1to6_vb, camsix_multi) # [2, 6, 512, 512]
        cams4 = torch.einsum('nchw,nchw->nchw', cam1to6_multi, camsix_vb) # [2, 6, 512, 512]
        cam_all = torch.cat([cams3, cams4],1) # [2, 12, 512, 512]
        cam_all = self.down(cam_all)
        cam = self.Show_Pseudo_Label(cam_all)

        return cam_all


          
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
                        
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

        # aff


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # attn_
        attn_ = attention_scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        #1
        attention_probs = self.attn_dropout(attention_probs)#[2, 12, 1024, 1024]

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        ##2
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        # aff
        # attn_ = attn_ + attn_.permute(0, 1, 3, 2)
        # print (attn_.shape) [2, 12, 1024, 1024]
        # print (hidden_states.shape) [2, 1024, 768]
        attn_copy = attn_.clone() # [2, 12, 1024, 1024]
        attn_copy = attn_copy.reshape(-1, self.num_attention_heads, attn_copy.shape[-1], attn_copy.shape[-1])

        return attention_output, weights, attn_copy
                        
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.config = config
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
        
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
    def forward(self, x):

        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        self.Eye_Center = Eye_Center(C, config)
        self.return_CAM = Class_Activation_Mapping()
        self.config = config

    def forward(self, x,complex_x):
        global Cam_Centre
        global Cam_correction_1
        global Cam_correction_2
        if x.shape[0] == 1:
            Cam_correction = Cam_correction_1
        else:
            Cam_correction = Cam_correction_2
        
        h = x # [2, 1024, 768]
        x = self.attention_norm(x) # [2, 1024, 768]
        x_E = x.view(x.shape[0], int(math.sqrt(x.shape[1])),int(math.sqrt(x.shape[1])),x.shape[2])
        x_eye_atten,attn = self.Eye_Center(x_E, Cam_Centre,complex_x) # [2, 32, 32, 768]
        x_eye_atten = x_eye_atten.view(x_eye_atten.shape[0], x_eye_atten.shape[1] * x_eye_atten.shape[2],x_eye_atten.shape[3])

        x, weights, _attns = self.attn(x) # [2, 1024, 768]
        # aff
        x_cam = x.permute(0, 2, 1).contiguous()
        x_cam = x.reshape(x_cam.shape[0], x_cam.shape[1], int(math.sqrt(x_cam.shape[2])), int(math.sqrt(x_cam.shape[2])))  # [2, 768, 32, 32]
        cam, cam_centre = self.return_CAM(x_cam, "vb", Cam_correction) # cam [2, 5, 32, 32], attn [2, 1024, 1024]
        Cam_correction = cam
        Cam_Centre = cam_centre
        if x.shape[0] == 1:
            Cam_correction_1 = Cam_correction
        else:
            Cam_correction_2 = Cam_correction

        x = h + x + 0.1* x_eye_atten # h[2, 1024, 768]
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
               
        return x, weights, cam, cam_centre

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")][0:self.config.hidden_size, 0:12, 0: int(self.config.hidden_size/12)]).contiguous().view(self.config.hidden_size, self.config.hidden_size).t()  # (768, 12, 64) -- > (hidden_size, 12, 64)
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")][0:self.config.hidden_size, 0:12, 0: int(self.config.hidden_size/12)]).contiguous().view(self.hidden_size, self.hidden_size).t()    # (768, 12, 64) -- > (hidden_size, 12, 64)
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")][0:self.config.hidden_size, 0:12, 0: int(self.config.hidden_size/12)]).contiguous().view(self.hidden_size, self.hidden_size).t()  # (768, 12, 64) -- > (hidden_size, 12, 64)
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")][0:12, 0: int(self.config.hidden_size/12), 0:self.config.hidden_size]).contiguous().view(self.hidden_size, self.hidden_size).t()  # (12, 64, 768) -- > (12, 64, hidden_size)

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")][0:12,0: int(self.config.hidden_size/12)]).contiguous().view(-1)    # (12, 64)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")][0:12,0: int(self.config.hidden_size/12)]).contiguous().view(-1)      # (12, 64)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")][0:12,0: int(self.config.hidden_size/12)]).contiguous().view(-1)    # (12, 64)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")][0:self.config.hidden_size]).contiguous().view(-1)    # (768) -- > (hidden_size)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)

            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)
            # Parameter adjustable
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")][0:self.config.hidden_size, 0:self.config.transformer.mlp_dim]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")][0:self.config.transformer.mlp_dim, 0:self.config.hidden_size]).t()
            #mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t() # [3072, 768]
            #mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t() # [768, 3072]
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")][0:self.config.transformer.mlp_dim]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")][0:self.config.hidden_size]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")][0:self.config.hidden_size])) # 768 --> (hidden_size)
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")][0:self.config.hidden_size]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")][0:self.config.hidden_size]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")][0:self.config.hidden_size]))
               

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        self.upcam =  nn.UpsamplingBilinear2d(scale_factor=16)
        self.return_avgCAM = Show_Avg_CAM_Mapping()

    def forward(self, hidden_states,complex_x):
        attns = []
        attn_weights = []
        cams = []
        hidden_get = []
        for layer_block in self.layer:
            hidden_states, weights, cam, attn = layer_block(hidden_states,complex_x)
            # print (attn.shape) [2, 12, 1024, 1024]
            attns.append(attn)
            cam = self.upcam(cam)
            cams.append(cam)
            hidden_get.append(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        cams = torch.stack(cams) # [6, 2, 6, 512, 512]
        camsone = torch.mean(cams, dim=2)  
        camsone = camsone.sum(0)
        camssix = torch.mean(cams, dim=0)
        camout = self.return_avgCAM(camsone, "vb")
        return encoded, attn_weights, camsone,camssix, attns,hidden_get
        
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids,complex_x):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights, camone, camsix, attns,hidden_get = self.encoder(embedding_output,complex_x)  # (B, n_patch, hidden)
        # aff
        return encoded, attn_weights, features, camone, camsix, attns,hidden_get

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = [1024, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]


        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous()
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)  #x[2, 512, 32, 32]

        for i, decoder_block in enumerate(self.blocks):        
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            
        return x

class DecoderBlock(nn.Module):
#change ganshouye
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        #print (1)
        #print(in_channels)
        #print(out_channels)
        
            
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        #print (2)
        #print(in_channels)
        #print(out_channels)
        
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        #print("x")
        #print(x.shape)
        #print (skip)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        #print("x")
        #print(x.shape)    
        
        x = self.conv2(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.channels = [512,256,128,3]
        self.dropout = nn.Dropout2d(p=0.5)
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(256,self.channels[0],3,1,1,bias=False),
                                    nn.BatchNorm2d(self.channels[0]),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(self.channels[0], self.channels[1], 4,2, 1, bias=False),
                                     nn.BatchNorm2d(self.channels[1]),
                                     nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(self.channels[1], self.channels[2], 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(self.channels[2]),
                                     nn.ReLU(inplace=True))

        self.layer4 = nn.ConvTranspose2d(self.channels[2], self.channels[3], 4, 2, 1, bias=False)

        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.layer4(out)
        out= self.tanh(out)

        return out





class Discriminator(nn.Module):
    """Discriminator model."""

    def __init__(self,num_class):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.num_class = num_class
        self.channels = [512, 1024, 2048, 1]
        self.dropout = nn.Dropout2d(p=0.5)
        self.layer1 = nn.Sequential(nn.Conv2d(256, self.channels[0], 1, 1, bias=False),
                                     nn.LeakyReLU(0.2,inplace=True))

        self.layer2 = nn.Sequential(nn.Conv2d(self.channels[0], self.channels[1], 3,1, 1, bias=False),
                                     nn.InstanceNorm2d(self.channels[1]),
                                     nn.LeakyReLU(0.2,inplace=True))

        self.layer3 = nn.Sequential(nn.Conv2d(self.channels[1], self.channels[2], 3,1, 1, bias=False),
                                     nn.InstanceNorm2d(self.channels[2]),
                                     nn.LeakyReLU(0.2,inplace=True))

        self.cls = nn.Conv2d(self.channels[2],self.num_class,3,1,padding=2,dilation=2,bias=False)

        self.layer4 = nn.Conv2d(self.channels[2]*2, self.channels[3], 3,1,1,bias=False)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.class_label = nn.Conv2d(self.channels[2],self.num_class,kernel_size=4,stride=1,padding=2, dilation=2, bias=False)

    def forward_onece(self,x):
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        return out

    def forward(self, x1,x2,domain = 'S'):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1,out2),1)
        d = self.layer4(d)
        d = self.sigmoid(d)
        if domain == 'S':
            c1 = self.cls(out1)
            c2 = self.cls(out2)
            return d,c1,c2
        else:
            return d




class Discriminator_NumClass_2(nn.Module):
    def __init__(self, num_class):
        super(Discriminator_NumClass_2, self).__init__()

        self.num_class = num_class
        self.channels = [64, 128, 256, 512, 1]
        self.layer1 = nn.Conv2d(self.num_class, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(self.channels[3]*2, self.channels[4], kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward_onece(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        out = self.leaky_relu(out)
        out = self.layer4(out)
        #print out.shape
        out = self.leaky_relu(out)
        # if input is 1 img, out is (512,32,32)
        return out

    def forward(self, x1, x2):
        """Forward the discriminator."""
        out1 = self.forward_onece(x1)
        out2 = self.forward_onece(x2)
        d = torch.cat((out1, out2), 1)
        # if input is 2 img, out is (1, 1, 16, 16 )

        d = self.layer5(d)
        d = self.sigmoid(d)
        return d

class SoftDiceLoss(nn.Module):
# dice_loss1 = dice_criterion(output_train[0,...], gt_onehot1).item()
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation='sigmoid', reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        #print y_pred.shape
        #print y_true.shape
    
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice


def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):


    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)
    #print pred
    #print gt.shape
    #H = pred.size(0)

    N = gt.size(0)
    #print N
    #print pred.shape
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat.cuda() * gt_flat.cuda()).sum(1)
    unionset = (pred_flat.cuda()).sum(1) + (gt_flat.cuda()).sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):


    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N



def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out






def Res_Deeplab(num_classes=21, is_refine=False):
    if is_refine:
        model = ResNet_Refine(Bottleneck,[3, 4, 23, 3], num_classes)
    else:
        model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model

def Get_Discrimer(input_size=(512,512), hidden_dims=512, output_dims=1):
    out_size=outS(input_size[0])
    input_dims = out_size*out_size*256
    model = Discriminator(input_dims=input_dims,hidden_dims=hidden_dims,output_dims=output_dims)
    return model

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
    'basical-transblock': configs.basical_transblock(),
}