import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .mm_grounding_dino import GroundingDINOVisionTower
from .grounding_dino_mixed import GroundingDINOMixedVisionTower
from .siglip_encoder import SigLipVisionTower

def build_vision_tower(vision_tower_cfg, ori_vision_tower=None, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower == 'grounding_dino':
        return GroundingDINOVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower == 'clip':
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower == 'grounding_dino_mixed':
        return GroundingDINOMixedVisionTower(vision_tower, ori_vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    raise ValueError(f'Unknown vision tower: {vision_tower}')
