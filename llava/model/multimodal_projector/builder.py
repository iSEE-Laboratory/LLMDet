import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class vision_projector_with_pos_proj(nn.Module):
    def __init__(self, hidden_size, vision_projector):
        super().__init__()
        self.vision_projector = nn.ModuleList(vision_projector)

        self.pos_proj = nn.Linear(256 * 2, hidden_size)

    def forward(self, x):
        out = []
        for module, a in zip(self.vision_projector, x):
            out.append(module(a))
        return torch.cat(out)

    def forward_pos(self, pos):
        return self.pos_proj(pos)
    


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    vision_tower = getattr(config, 'mm_vision_tower', getattr(config, 'vision_tower', None))
    if 'grounding_dino_mixed' == vision_tower:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size + 256, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            vision_projector =  nn.Sequential(*modules)

        return vision_projector
        

    if projector_type == 'linear':
        vision_projector =  nn.Linear(config.mm_hidden_size, config.hidden_size)
        if not getattr(config, 'plain_projector', False) and 'grounding_dino' in vision_tower:
            return vision_projector_with_pos_proj(config.hidden_size, [vision_projector])
        else:
            return vision_projector

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        vision_projector =  nn.Sequential(*modules)
        if not getattr(config, 'plain_projector', False) and ('grounding_dino' in vision_tower or getattr(config, 'load_full_model', False)):
            return vision_projector_with_pos_proj(config.hidden_size, [vision_projector])
        else:
            return vision_projector

    if projector_type == 'identity':
        vision_projector = IdentityMap()
        if not getattr(config, 'plain_projector', False) and 'grounding_dino' in vision_tower:
            return vision_projector_with_pos_proj(config.hidden_size, [vision_projector])
        else:
            return vision_projector

    raise ValueError(f'Unknown projector type: {projector_type}')
