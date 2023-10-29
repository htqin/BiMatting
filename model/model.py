import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
from .mobilenet import BiMobileNet
from .decoder import Projection
from .fast_guided_filter import FastGuidedFilterRefiner

from .bidecoder import BiRecurrentDecoder
from .lraspp import BiLRASPP
from .deep_guided_filter import DeepGuidedFilterRefiner, BiDeepGuidedFilterRefiner

"""
Adopted from <https://github.com/PeterL1n/RobustVideoMatting>
"""

class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['bimobilenet']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        self.variant = variant
        
        if variant == 'bimobilenet':
            self.backbone = BiMobileNet(pretrained_backbone) # Binary
            self.aspp = BiLRASPP(1024, 128)
            self.decoder = BiRecurrentDecoder([16, 32, 64, 128], [80, 40, 32, 16])
               
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            if variant in ['bimobilenet']:
                self.refiner = BiDeepGuidedFilterRefiner()
            else:
                self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        
        hid, hid_s, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4, self.project_mat)
    
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            fgr_residual_s, pha_s = self.project_mat(hid_s).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            fgr_s = fgr_residual_s + F.interpolate(src.flatten(0,1), size=fgr_residual_s.shape[-2:], mode='bilinear', align_corners=False)
            fgr_s = fgr_s.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            pha_s = pha_s.clamp(0., 1.)
            return [fgr, pha, fgr_s, pha_s, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]       

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
