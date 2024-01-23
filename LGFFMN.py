import pywt
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
    

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out',avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out',max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        # print('cat',x.shape)
        x = self.conv1(x)
        # print('conv',x.shape)
        y = self.sigmoid(x)
        # print('sigmoid',y.shape)
        return y * res


class CA_Block2D(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CA_Block2D, self).__init__()
      
        # self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        # self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channel // reduction)

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=mip, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(mip)

        self.F_h = nn.Conv2d(mip, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(mip, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.avg_pool_x(x)
        # print(x_h.size())
        x_w = self.avg_pool_y(x).permute(0, 1, 3, 2)
        # print(x_w.size())
     
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), dim=2)))
        # print(x_cat_conv_relu.size())

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], dim=2)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h))
        # print(s_h.size())
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w.permute(0, 1, 3, 2)))
        # print(s_w.size())
     
        out = x  * s_h * s_w
        # print(out.size())

        return out


class FRDAB(nn.Module):
    def __init__(self, n_feats=32):
        super(FRDAB, self).__init__()

        self.c1 = default_conv(n_feats, n_feats, 1)
        self.c2 = default_conv(n_feats, n_feats // 2, 3)
        self.c3 = default_conv(n_feats, n_feats // 2, 3)
        self.c4 = default_conv(n_feats*2, n_feats, 3)
        self.c5 = default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = default_conv(n_feats*2, n_feats, 1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        y1 = self.act(self.c1(x))
        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))
        cat1 = torch.cat([y1, y2, y3], 1)

        y4 = self.act(self.c4(cat1))
        y5 = self.c5(y3)  # 16
        cat2 = torch.cat([y2, y5, y4], 1)

        y6 = cat2
        y7 = self.c6(y6)

        output = res + y7
        return output


class LFFM(nn.Module):
    def __init__(self, n_feats=32):
        super(LFFM, self).__init__()

        self.b1 = FRDAB(n_feats=n_feats)
        self.b2 = FRDAB(n_feats=n_feats)
        self.b3 = FRDAB(n_feats=n_feats)

        self.c1 = nn.Conv2d(2 * n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        self.c2 = nn.Conv2d(3 * n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        self.c3 = nn.Conv2d(4 * n_feats, n_feats, 1, stride=1, padding=0, groups=1)

    def forward(self, x):
        res = x

        out1 = self.b1(x)
        dense1 = torch.cat([x, out1], 1)

        out2 = self.b2(self.c1(dense1))

        dense2 = torch.cat([x, out1, out2], 1)
        out3 = self.b3(self.c2(dense2))

        dense3 = torch.cat([x, out1, out2, out3], 1)
        out4 = self.c3(dense3)

        output = res + out4

        return output


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )
        
        # self.ca = CALayer(dim)
        # self.ca = CA_Block2D(dim)

        

    def forward(self, x):
        # x = self.ccm(x)
        # x = self.ca(x)
        # return x
        return self.ccm(x)


# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        
        # Activation
        self.act = nn.GELU() 

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out



class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.safm = SAFM(dim) 
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale) 

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class Inta(nn.Module):
    def __init__(self,dim):
        super(Inta,self).__init__()
        
        self.Mix = nn.Sequential(
                nn.Conv2d(dim*3,dim//2,kernel_size=3,padding=1),
                nn.Conv2d(dim//2,dim,kernel_size=3,padding=1)
            )          
        self.error_resblock = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1),
        )
    def forward(self,xl,xi,xn):
        
        h1 = xi
        h2 = self.Mix(torch.cat([xl,xi,xn],dim=1))
        e = h2 - h1
        e = self.error_resblock(e)
        h1 = h1 + e
        return h1
    
  
# class Inta(nn.Module):
#     def __init__(self,dim,scale):
#         super(Inta,self).__init__()
          
#         TwoTail = []
#         if (scale & (scale - 1)) == 0: 
#             for _ in range(int(math.log(scale, 2))):
#                 TwoTail.append(nn.Conv2d(dim, dim*4, kernel_size=(3,3), stride=1, padding=(1,1)))
#                 TwoTail.append(nn.PixelShuffle(2))           
#         else:
#             TwoTail.append(nn.Conv2d(dim, dim*9, kernel_size=(3,3), stride=1, padding=(1,1)))
#             TwoTail.append(nn.PixelShuffle(3))  

#         # TwoTail.append(nn.Conv2d(dim, out_channels, kernel_size=(3,3),  stride=1, padding=(1,1)))                               	    	
#         self.TwoTail = nn.Sequential(*TwoTail)

#         self.Mix = nn.Sequential(
#                 nn.Conv2d(dim*3,dim//2,kernel_size=3,padding=1),
#                 nn.Conv2d(dim//2,dim,kernel_size=3,padding=1)
#             )          
        
#         self.error_resblock = nn.Sequential(
#             nn.Conv2d(dim,dim,kernel_size=3,padding=1),
#         )
#     def forward(self,xl,xi,xn):
        
#         h1 = self.TwoTail(xi)
#         h2 = self.TwoTail(self.Mix(torch.cat([xl,xi,xn],dim=1)))
#         e = h2 - h1
#         e = self.error_resblock(e)
#         h1 = h1 + e
#         return h1  
    
    
class LGFFM(nn.Module):
    def __init__(self,n_subs=8,nfeats=128,n_blocks=2,ffn_scale=2.0, scale=4):
            
        super(LGFFM, self).__init__()

        self.n_blocks = n_blocks
        self.nfeats = nfeats
    
        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        head = []
        head.append(nn.Conv2d(n_subs, nfeats, kernel_size=(3,3), stride=1, padding=(1,1)))
        # head.append(nn.Conv2d(nfeats, nfeats, kernel_size=(1,1), stride=1, padding=(0,0)))
        self.head = nn.Sequential(*head)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        
        self.r1 = LFFM(n_feats=nfeats)
        self.r2 = LFFM(n_feats=nfeats)
        self.r3 = LFFM(n_feats=nfeats)

        
        self.c1 = default_conv(3*nfeats, nfeats, 1)
        self.c2 = default_conv(nfeats, nfeats, 3)

        # self.c1 = default_conv(nfeats, nfeats, 1)
        # self.c2 = default_conv(nfeats, nfeats, 3)
        # self.c3 = default_conv(nfeats, nfeats, 3)
        
        self.feats = nn.Sequential(*[AttBlock(nfeats, ffn_scale) for _ in range(n_blocks)])
        
        #####################################################################################################
        ################################### 3, high quality image reconstruction #####################################
        
        
        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append((nn.Conv2d(nfeats, nfeats*4, kernel_size=(3,3), stride=1, padding=(1,1))))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(nn.Conv2d(nfeats, nfeats*9, kernel_size=(3,3), stride=1, padding=(1,1)))
            TwoTail.append(nn.PixelShuffle(3))  

        self.TwoTail = nn.Sequential(*TwoTail)
        

    def forward(self, x):
        x = self.head(x)
        res = x
        
        y1 = self.r1(x)
        y2 = self.r2(y1)
        y3 = self.r3(y2)
        y4 = torch.cat([y1, y2, y3], dim=1)
        # y4 = torch.cat([y1, y2], dim=1)
        # y4 = y1
        # y4 = x
        
        y4 = self.c1(y4)
        y5 = self.c2(y4)
        
        restore = self.feats(y5)

        # # no SAFMN
        # restore = y5
        
        restore = res + restore
        restore = self.TwoTail(restore)
                
        return  restore


class LGFFMN(nn.Module):
    def __init__(self,
        nfeats=128,
        n_blocks=4,
        n_colors=31,
        ffn_scale=2.0,
        n_subs=8,
        n_ovls=2,
        scale=4):
        super(LGFFMN,self).__init__()

        # calculate the group number (the number of branch networks)
        # 向上取整计算组的数量 G 
        self.n_subs = n_subs
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []        
        self.scale = scale
        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            # 把每一组的开始 idx 与结束 idx 存入 list
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.branch = LGFFM(n_subs, nfeats, n_blocks, ffn_scale, scale)
        
        # inta
        # self.branch_up = Inta(nfeats)

        self.branch_recon = nn.Conv2d(nfeats, n_subs, kernel_size=3,padding=1)

    def forward(self, x, bicu):
    
        b, c, h, w = x.shape
        m = []
        y = torch.zeros(b, c,  h*self.scale,  w*self.scale).cuda() 
        # y = torch.zeros(b, c,  h*self.scale,  w*self.scale) 

        channel_counter = torch.zeros(c).cuda()    
        # channel_counter = torch.zeros(c)   

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            xi = self.branch(xi)
            m.append(xi)
            
        # for g in range(self.G):
        #     sta_ind = self.start_idx[g]
        #     end_ind = self.end_idx[g]

        #     if g==0:
        #         xl = m[g+2]
        #         xi = m[g]
        #         xn = m[g+1]
        #     elif g==self.G-1:
        #         xl = m[g-1]
        #         xi = m[g]
        #         xn = m[g-2]
        #     else:
        #         xl = m[g-1]
        #         xi = m[g]
        #         xn = m[g+1]  

            # inta
            # xi = self.branch_up(xl,xi,xn)

            xi = self.branch_recon(xi)
            y[:, sta_ind:end_ind, :, :] += xi
            # 用 channel_counter 记录某一个位置被加了几次，然后再除这个数字取平均
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral in
        # dices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = y + bicu
        return y   
