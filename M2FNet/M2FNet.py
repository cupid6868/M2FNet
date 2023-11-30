import torch
import math
from torch.nn import LayerNorm, Linear, Dropout, Softmax
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
import numpy as np
from auxil import yaml_load
from torchsummary import summary

tct = yaml_load('./cfg.yaml')
FM = 16
BATCH_SIZE_TRAIN = 1
NUM_CLASS = 0

if tct['train_dataset'] == 'trento':
    NUM_CLASS = 6
elif tct['train_dataset'] == 'HOUSTON2013':
    NUM_CLASS = 15
elif tct['train_dataset'] == 'aug':
    NUM_CLASS = 7




def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

def conv_orth_dist(kernel, stride=1):
    [o_c, i_c, w, h] = kernel.shape
    assert (w == h), "Do not support rectangular kernel"
    # half = np.floor(w/2)
    assert stride < w, "Please use matrix orthgonality instead"
    new_s = stride * (w - 1) + w  # np.int(2*(half+np.floor(half/stride))+1)
    temp = torch.eye(new_s * new_s * i_c).reshape((new_s * new_s * i_c, i_c, new_s, new_s)).cuda()
    out = (F.conv2d(temp, kernel, stride=stride)).reshape((new_s * new_s * i_c, -1))
    Vmat = out[np.floor(new_s ** 2 / 2).astype(int)::new_s ** 2, :]
    temp = np.zeros((i_c, i_c * new_s ** 2))
    for i in range(temp.shape[0]): temp[i, np.floor(new_s ** 2 / 2).astype(int) + new_s ** 2 * i] = 1
    return torch.norm(Vmat @ torch.t(out) - torch.from_numpy(temp).float().cuda())

def orth_dist(mat, stride=None):
    mat = mat.reshape((mat.shape[0], -1))
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1, 0)
    return torch.norm(torch.t(mat) @ mat - torch.eye(mat.shape[1]).cuda())

class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)
        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))
        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)
        if self.type == 'erosion2d':
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError
        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # (B, Cout, L)
        if self.type == 'erosion2d': x = -1 * x
        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)
        return x

class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')

class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')

class SpectralMorph(nn.Module):
    def __init__(self, FM, NC, kernel=3):
        super(SpectralMorph, self).__init__()

        self.erosion = Erosion2d(NC, FM, kernel, soft_max=False)
        self.conv1 = nn.Conv2d(FM, FM, 1, padding=0)
        self.dilation = Dilation2d(NC, FM, kernel, soft_max=False)
        self.conv2 = nn.Conv2d(FM, FM, 1, padding=0)

    def forward(self, x):
        z1 = self.erosion(x)
        z1 = self.conv1(z1)
        z2 = self.dilation(x)
        z2 = self.conv2(z2)
        return z1 + z2

class SpatialMorph(nn.Module):
    def __init__(self, FM, NC, kernel=3):
        super(SpatialMorph, self).__init__()
        self.erosion = Erosion2d(NC, FM, kernel, soft_max=False)
        self.conv1 = nn.Conv2d(FM, FM, 3, padding=1)
        self.dilation = Dilation2d(NC, FM, kernel, soft_max=False)
        self.conv2 = nn.Conv2d(FM, FM, 3, padding=1)

    def forward(self, x):
        z1 = self.erosion(x)
        z1 = self.conv1(z1)
        z2 = self.dilation(x)
        z2 = self.conv2(z2)
        return z1 + z2

class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
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


class Block(nn.Module):
    def __init__(self, dim, blockNum=0):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        kernels = [3, 5]
        self.cls_norm = LayerNorm(dim, eps=1e-6)
        self.spec_morph = nn.Sequential(SpectralMorph(FM, FM * 2, kernels[blockNum]), nn.BatchNorm2d(FM), nn.GELU())
        self.spat_morph = nn.Sequential(SpatialMorph(FM, FM * 2, kernels[blockNum]), nn.BatchNorm2d(FM), nn.GELU())

    def forward(self, x):
        ht, w = x.shape[2:]
        rest = x[:, 1:]
        rest1 = rest
        rest1 = self.spec_morph(rest1)
        rest2 = rest
        rest2 = self.spat_morph(rest2)
        rest = torch.cat([rest1, rest2], dim=1)
        x = torch.cat([x[:, 0:1, :], rest], dim=1)
        clsTok = x[:, 0:1]
        h = clsTok
        clsTok = self.attn(self.attention_norm(x.reshape(x.shape[0], x.shape[1], -1))).reshape(x.shape[0], 1, ht, w)
        clsTok = clsTok + h
        clsTok = self.cls_norm(clsTok.reshape(clsTok.shape[0], clsTok.shape[1], -1)).reshape(clsTok.shape)
        x = torch.cat([clsTok, x[:, 1:]], dim=1)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

class DConvformerEncoder(nn.Module):
    def __init__(
            self,
            *,
            dim,
            FM,
            dropout=0.1
    ):
        super().__init__()
        self.encode = Dilated_convformer(dim,dim,FM)

    def forward(self, h_tokens, l_tokens):
        h_tokens, l_tokens = self.encode(h_tokens, l_tokens)
        return h_tokens, l_tokens

class test(nn.Module):
    def __init__(self, ):
        super().__init__()


    def forward(self, x):
        return x

class Dilated_convformer(nn.Module):
    def __init__(self, dim, mlp_dim, FM, dropout=0.1):
        super().__init__()
        self.FM = FM
        kernels = [3, 5]
        self.spec_morph = nn.Sequential(SpectralMorph(FM, FM * 2, kernels[0]), nn.BatchNorm2d(FM), nn.GELU())
        self.spat_morph = nn.Sequential(SpatialMorph(FM, FM * 2, kernels[0]), nn.BatchNorm2d(FM), nn.GELU())
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, ConvModule())),
                # Residual(LayerNormalize(dim, Attention(dim=dim))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x0, x1):
        x0 = x0.reshape(x0.shape[0], x0.shape[1], int(math.sqrt(self.FM * 4)), int(math.sqrt(self.FM * 4)))
        x1 = x1.reshape(x1.shape[0], x1.shape[1], int(math.sqrt(self.FM * 4)), int(math.sqrt(self.FM * 4)))
        rest_x0 = x0[:, 1:]
        rest_x1 = x1[:, 1:]

        #hsi分支
        rest_x0_1 = rest_x0
        rest_x0_1 = self.spat_morph(rest_x0_1)
        rest_x0_2 = rest_x0
        rest_x0_2 = self.spec_morph(rest_x0_2)
        rest_x0 = torch.cat([rest_x0_1, rest_x0_2], dim=1)
        x0 = torch.cat([x0[:, 0:1, :], rest_x0], dim=1)
        x0 = rearrange(x0, 'b c h w  -> b c ( h w )')

        #lidar分支
        rest_x1_1 = rest_x1
        rest_x1_1 = self.spat_morph(rest_x1_1)
        rest_x1_2 = rest_x1
        rest_x1_2 = self.spec_morph(rest_x1_2)
        rest_x1 = torch.cat([rest_x1_1, rest_x1_2], dim=1)
        x1 = torch.cat([x1[:, 0:1, :], rest_x1], dim=1)
        x1 = rearrange(x1, 'b c h w  -> b c ( h w )')

        for Dilated_conv, mlp in self.layers:
            x0 = Dilated_conv(x0)
            x1 = Dilated_conv(x1)
            (h_cls, h_patch_tokens), (l_cls, l_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (x0, x1))
            x0 = torch.cat((l_cls, h_patch_tokens), dim=1)
            x1 = torch.cat((h_cls, l_patch_tokens), dim=1)

            x0, x1 = mlp(x0), mlp(x1)

        # x0 = x0.reshape(x0.shape[0],-1)
        # x1 = x1.reshape(x1.shape[0], -1)

        return x0, x1

class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        # 定义六层卷积层
        # 两层HDC（1,2,5,1,2,5）
        self.conv = nn.Sequential(
            # 第一层 (3-1)*1+1=3 （64-3)/1 + 1 =62
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(8),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第二层 (3-1)*2+1=5 （62-5)/1 + 1 =58
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第三层 (3-1)*5+1=11  (58-11)/1 +1=48
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第四层(3-1)*1+1=3 （48-3)/1 + 1 =46
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(16),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第五层 (3-1)*2+1=5 （46-5)/1 + 1 =42
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(8),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第六层 (3-1)*5+1=11  (42-11)/1 +1=32
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(1),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True)
        )
        # 输出层,将通道数变为分类数量

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv(x)
        x = torch.squeeze(x, dim=1)
        return x

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_0=3, kernel_size_1=5, stride=1, padding=None, bias=None,
                 p=64, g=64):
        super(HetConv, self).__init__()
        # Groupwise Convolution kernel_size=3
        self.gwc1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_0, groups=g,
                              padding=kernel_size_0 // 3, stride=stride)

        # Groupwise Convolution kernel_size=5
        self.gwc2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_1, groups=g, padding=5 // 2, )

        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)

    def forward(self, x):
        a = self.gwc1(x)
        b = self.gwc2(x)
        c = self.pwc(x)
        return a + b + c

class feature_tokens(nn.Module):
    def __init__(self, FM):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, FM * 4))
        self.token_wA = nn.Parameter(torch.empty(1, FM * 2, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)
        self.position_embeddings = nn.Parameter(torch.zeros(1, FM * 2 + 1, FM * 4))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        wa = self.token_wA.expand(x.shape[0], -1, -1)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x.shape[0], -1, -1)
        VV = torch.einsum('bij,bjk->bik', x, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        x = torch.cat((cls_tokens, T), dim=1)  # [b,n+1,dim]
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class M2Fnet(nn.Module):
    def __init__(
            self,
            FM, NC, Classes,
            dropout=0.1,

    ):
        super(M2Fnet, self).__init__()

        self.FM = FM
        self.conv1_HSI = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv1_LiDAR = nn.Sequential(
            nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Multi_Scale_conv2_HSI = nn.Sequential(
            HetConv(8 * (NC - 8), FM * 4,
                    p=1,
                    g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
                    ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU()
        )
        self.Multi_Scale_conv2_LiDAR = nn.Sequential(
            HetConv(64, FM * 4,
                    p=1,
                    g=(FM * 4) // 2 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 4,
                    ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU()
        )
        self.feature_tokens = feature_tokens(FM)
        self.DConvformer_encoder = DConvformerEncoder(
            dim=FM * 4,
            dropout=dropout,
            FM=FM,
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, Classes))




    def forward(self, x1, x2, mask=None):
        # backbone
        x1 = x1.reshape(x1.shape[0], -1, tct['patch_size'], tct['patch_size'])
        x1 = x1.unsqueeze(1)
        x2 = x2.reshape(x2.shape[0], -1, tct['patch_size'], tct['patch_size'])

        x1 = self.conv1_HSI(x1)
        x1 = x1.reshape(x1.shape[0], -1, tct['patch_size'], tct['patch_size'])
        x2 = self.conv1_LiDAR(x2)
        x2 = x2.reshape(x2.shape[0], -1, tct['patch_size'], tct['patch_size'])

        x1 = self.Multi_Scale_conv2_HSI(x1)
        x2 = self.Multi_Scale_conv2_LiDAR(x2)

        # feature-->tokens
        x1 = self.feature_tokens(x1)
        x2 = self.feature_tokens(x2)

        # DConvformer
        x1, x2 = self.DConvformer_encoder(x1, x2)
        x1, x2, = map(lambda t: t[:, 0], (x1, x2))

        Classifier_Head = self.mlp_head(x1) + self.mlp_head(x2)

        return Classifier_Head

if __name__ == '__main__':
    FM = 16
    NC = 144
    Classes = 15
    model = M2Fnet(FM, NC, Classes)
    model.eval()
    print(model)
    input1 = torch.randn(64, 144, 121)
    input2 = torch.randn(64, 1, 121)
    x = model(input1, input2)
    print(x.size())
    # summary(model, [(64, 30, 121), (64, 1, 121)],device='cpu')
