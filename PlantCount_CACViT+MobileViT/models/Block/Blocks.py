import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x ,attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x_1,attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTLikeBlock(nn.Module):
    def __init__(self, dim, num_heads, num_image_patches, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), expansion=4):
        super().__init__()
        self.num_image_patches = num_image_patches
        self.conv = MV2Block(inp=dim, oup=dim, stride=1, expansion=expansion)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x_y):
        x = x_y[:, :self.num_image_patches, :]
        y = x_y[:, self.num_image_patches:, :]
        
        B, N_img, D = x.shape
        H = W = int(N_img**0.5)
        
        x_conv = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        x_conv = self.conv(x_conv)
        x_conv = rearrange(x_conv, 'b d h w -> b (h w) d')
        
        x = x + x_conv
        
        x_y = torch.cat((x, y), dim=1)
        
        x_1,attn = self.attn(self.norm1(x_y))
        x_y = x_y + self.drop_path(x_1)
        x_y = x_y + self.drop_path(self.mlp(self.norm2(x_y)))
        return x_y, attn


# ======= 新增：真正的MobileViT Block =======
class MobileViTAttention(nn.Module):
    """MobileViT的轻量级注意力机制"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out), attn


class MobileViTTransformer(nn.Module):
    """MobileViT的Transformer模块"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                MobileViTAttention(dim, heads, dim_head, dropout),
                nn.LayerNorm(dim),
                Mlp(dim, mlp_dim, dim, drop=dropout)
            ]))
    
    def forward(self, x):
        attn_maps = []
        for norm1, attn, norm2, ff in self.layers:
            attn_output, attn_map = attn(norm1(x))
            x = attn_output + x
            x = ff(norm2(x)) + x
            attn_maps.append(attn_map)
        return x, attn_maps


class MobileViTBlock(nn.Module):
    """真正的MobileViT Block，结合CNN和Transformer"""
    def __init__(self, dim, num_heads, num_image_patches, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), expansion=4, 
                 kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        self.num_image_patches = num_image_patches
        self.ph, self.pw = patch_size
        
        # 局部特征提取 (CNN部分)
        self.conv1 = conv_nxn_bn(dim, dim, kernel_size)
        self.conv2 = conv_1x1_bn(dim, dim)
        
        # 全局特征建模 (Transformer部分)
        self.transformer = MobileViTTransformer(dim, depth=2, heads=num_heads, 
                                               dim_head=dim//num_heads, 
                                               mlp_dim=int(dim * mlp_ratio), 
                                               dropout=drop)
        
        # 特征融合
        self.conv3 = conv_1x1_bn(dim, dim)
        self.conv4 = conv_nxn_bn(2 * dim, dim, kernel_size)
        
    def forward(self, x_y):
        x = x_y[:, :self.num_image_patches, :]
        y = x_y[:, self.num_image_patches:, :]
        
        B, N_img, D = x.shape
        H = W = int(N_img**0.5)
        
        # 保存原始特征用于残差连接
        y_original = y.clone()
        
        # 局部特征提取 (CNN)
        x_conv = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        x_conv = self.conv1(x_conv)
        x_conv = self.conv2(x_conv)
        
        # 全局特征建模 (Transformer)
        x_conv = rearrange(x_conv, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', 
                          ph=self.ph, pw=self.pw, h=H//self.ph, w=W//self.pw)
        x_conv, attn_maps = self.transformer(x_conv)
        x_conv = rearrange(x_conv, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', 
                          h=H//self.ph, w=W//self.pw, ph=self.ph, pw=self.pw)
        
        # 特征融合
        x_conv = self.conv3(x_conv)
        x_conv = rearrange(x_conv, 'b d h w -> b (h w) d')
        
        # 残差连接
        x = x + x_conv
        
        # 重新组合x和y
        x_y = torch.cat((x, y), dim=1)
        
        return x_y, attn_maps


# ======= 新增：轻量级MobileViT Block (用于解码器) =======
class LightweightMobileViTBlock(nn.Module):
    """轻量级MobileViT Block，用于解码器"""
    def __init__(self, dim, num_heads, num_image_patches, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_image_patches = num_image_patches
        
        # 简化的局部特征提取
        self.conv1 = conv_1x1_bn(dim, dim)
        
        # 轻量级Transformer
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x_y):
        x = x_y[:, :self.num_image_patches, :]
        y = x_y[:, self.num_image_patches:, :]
        
        B, N_img, D = x.shape
        H = W = int(N_img**0.5)
        
        # 局部特征提取
        x_conv = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        x_conv = self.conv1(x_conv)
        x_conv = rearrange(x_conv, 'b d h w -> b (h w) d')
        
        # 残差连接
        x = x + x_conv
        
        # 重新组合并应用Transformer
        x_y = torch.cat((x, y), dim=1)
        
        x_1, attn = self.attn(self.norm1(x_y))
        x_y = x_y + self.drop_path(x_1)
        x_y = x_y + self.drop_path(self.mlp(self.norm2(x_y)))
        
        return x_y, attn

# ======= 新增：真正轻量级MobileViT Block =======
class UltraLightweightMobileViTBlock(nn.Module):
    """超轻量级MobileViT Block，参数量最少"""
    def __init__(self, dim, num_heads, num_image_patches, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), reduction_ratio=4):
        super().__init__()
        self.num_image_patches = num_image_patches
        self.reduction_ratio = reduction_ratio
        
        # 使用深度可分离卷积减少参数量
        self.conv1 = nn.Sequential(
            # Depthwise卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            # Pointwise卷积 (降维)
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // reduction_ratio),
            nn.SiLU(),
            # Pointwise卷积 (升维)
            nn.Conv2d(dim // reduction_ratio, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
        # 轻量级Transformer (减少MLP维度)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # 减少MLP的隐藏维度
        mlp_hidden_dim = int(dim * mlp_ratio // 2)  # 减少一半
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x_y):
        x = x_y[:, :self.num_image_patches, :]
        y = x_y[:, self.num_image_patches:, :]
        
        B, N_img, D = x.shape
        H = W = int(N_img**0.5)
        
        # 局部特征提取 (深度可分离卷积)
        x_conv = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        x_conv = self.conv1(x_conv)
        x_conv = rearrange(x_conv, 'b d h w -> b (h w) d')
        
        # 残差连接
        x = x + x_conv
        
        # 重新组合并应用Transformer
        x_y = torch.cat((x, y), dim=1)
        
        x_1, attn = self.attn(self.norm1(x_y))
        x_y = x_y + self.drop_path(x_1)
        x_y = x_y + self.drop_path(self.mlp(self.norm2(x_y)))
        
        return x_y, attn


# ======= 新增：最轻量级版本 (仅用于解码器) =======
class MinimalMobileViTBlock(nn.Module):
    """最轻量级MobileViT Block，参数量最少"""
    def __init__(self, dim, num_heads, num_image_patches, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_image_patches = num_image_patches
        
        # 仅使用1x1卷积进行通道混合，参数量最少
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),  # 降维
            nn.BatchNorm2d(dim // 4),
            nn.SiLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),  # 升维
            nn.BatchNorm2d(dim)
        )
        
        # 标准Transformer (但减少MLP维度)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # 大幅减少MLP的隐藏维度
        mlp_hidden_dim = int(dim * mlp_ratio // 4)  # 减少到1/4
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x_y):
        x = x_y[:, :self.num_image_patches, :]
        y = x_y[:, self.num_image_patches:, :]
        
        B, N_img, D = x.shape
        H = W = int(N_img**0.5)
        
        # 局部特征提取 (1x1卷积)
        x_conv = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        x_conv = self.conv1(x_conv)
        x_conv = rearrange(x_conv, 'b d h w -> b (h w) d')
        
        # 残差连接
        x = x + x_conv
        
        # 重新组合并应用Transformer
        x_y = torch.cat((x, y), dim=1)
        
        x_1, attn = self.attn(self.norm1(x_y))
        x_y = x_y + self.drop_path(x_1)
        x_y = x_y + self.drop_path(self.mlp(self.norm2(x_y)))
        
        return x_y, attn

