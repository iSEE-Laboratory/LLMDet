import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import register_all_modules
register_all_modules()
from mmengine.config import Config
from mmdet.registry import MODELS
from mmengine.structures import BaseDataElement
from mmcv.ops.nms import nms
from mmdet.datasets.transforms import FixScaleResize, PackDetInputs
import numpy as np
import PIL.Image as Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from ram.models import ram_plus
from ram import get_transform
from .siglip_encoder import SigLipVisionTower

class GroundingDINOMixedVisionTower(nn.Module):
    def __init__(self, vision_tower, ori_vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # load openai clip
        if ori_vision_tower is None:
            ori_vision_tower = SigLipVisionTower('../huggingface/siglip-so400m-patch14-384/', None)
            self.clip_image_processor = ori_vision_tower.image_processor
            self.clip_vision_tower = ori_vision_tower
            self.clip_vision_tower.requires_grad_(False)
            delay_load = False
        else:
            self.clip_image_processor = ori_vision_tower.image_processor
            self.clip_vision_tower = ori_vision_tower
            self.clip_vision_tower.requires_grad_(False)

        self.grounding_dino_config = args.grounding_dino_config
        self.vision_tower_weight_path = args.vision_tower_weight_path.split('+')
        self.load_ram = args.load_ram
        self.bert_base_path = args.bert_base_path
        # assert len(self.vision_tower_weight_path) == 2
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.transform1 = FixScaleResize(scale=(800, 1333), keep_ratio=True, backend='pillow')
        self.transform2 = PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = Config.fromfile(self.grounding_dino_config)
            self.cfg_only.model.data_preprocessor.bgr_to_rgb = False
            self.cfg_only.model.language_model.name = self.bert_base_path
            self.cfg_only.model.test_cfg.chunked_size = -1
            self.cfg_only.model.lmm = None
        self.img_idx = 0


    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        # load grounding dino
        cfg = Config.fromfile(self.grounding_dino_config)
        cfg.model.data_preprocessor.bgr_to_rgb = False
        cfg.model.language_model.name = self.bert_base_path
        cfg.model.test_cfg.chunked_size = -1
        cfg.model.lmm = None
        self.vision_tower = MODELS.build(cfg.model)
        self.vision_tower.cfg = cfg
        checkpoint = torch.load(self.vision_tower_weight_path[0], 'cpu')
        msg = self.vision_tower.load_state_dict(checkpoint['state_dict'], False)
        print(msg)
        self.vision_tower.eval()
        self.vision_tower.to(device_map)

        self.image_processor = self.vision_tower.data_preprocessor
        self.vision_tower.requires_grad_(False)

        if self.load_ram:
            self.ram = ram_plus(pretrained=self.vision_tower_weight_path[1],
                    image_size=384,
                    vit='swin_l',
                    bert_base_path=self.bert_base_path)
            self.ram = self.ram.to(device_map)
            self.ram.eval()
            self.ram_transform = get_transform()
            self.ram.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        elif self.select_feature == 'cls':
            image_features = image_features[:, :1]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, tags):
        self.vision_tower.eval()
        self.clip_vision_tower.eval()
        if type(images) is not list:
            images = [images]

        if tags is None or tags[0] is None:
            ram_input_image = [self.ram_transform(img) for img in images]
            ram_input_image = torch.stack(ram_input_image, dim=0)
            ram_input_image = ram_input_image.to(self.device).to(self.dtype)
            self.ram.eval()
            new_tags, _ = self.ram.generate_tag(ram_input_image)
            tags = []
            for tag in new_tags:
                tag = tag.split(' | ') + ['text', 'symbol']
                tags.append('. '.join(tag) + '.')


        assert type(images) is list
        transformed_images = []
        img_shapes = []
        for image, tag in zip(images, tags):
            image = image.convert('RGB')
            img = np.array(image)
            img_shapes.append(img.shape[:2])
            results = {}
            results['img_path'] = None
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
            results['text'] = tag
            results['custom_entities'] = True
            results['tokens_positive'] = None
            results = self.transform1.transform(results)
            results = self.transform2.transform(results)
            transformed_images.append(results)

        batched_results = {}
        batched_results['inputs'] = [results['inputs'].to(self.device) for results in transformed_images]
        batched_results['data_samples'] = [BaseDataElement(metainfo={k:v for k,v in results.items() if k != 'inputs'}) for results in transformed_images]
        self.image_processor.to(self.device)
        batched_results = self.image_processor(batched_results)
        data_samples = []
        for res in batched_results['data_samples']:
            x = res.data_samples
            x.batch_input_shape = res.batch_input_shape
            data_samples.append(x)

        grounding_dino_output = self.vision_tower.predict_encoder_only(batched_results['inputs'].to(self.dtype), data_samples, False)

        # get clip feature
        processor = self.clip_image_processor
        image_tensors = []
        for image in images: # inplace update
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensors.append(image)
        images = torch.stack(image_tensors)
        clip_image_features = self.clip_vision_tower(images.to(device=self.device, dtype=self.dtype))
        # clip_image_features = self.feature_select(image_forward_outs).to(device=self.device, dtype=self.dtype)

        # merge features
        merged_image_features = []
        for dino_feat, clip_feat in zip(grounding_dino_output, clip_image_features):
            dino_feat_padded = torch.zeros((dino_feat.shape[0], 27, 27), dtype=dino_feat.dtype, device=dino_feat.device)
            height, width = dino_feat.shape[1:]
            if width == height:
                dino_feat_padded = F.interpolate(dino_feat[None], size=(27, 27), mode='bilinear')[0]
            elif width > height:
                new_height = int(height / width * 27)
                dino_feat = F.interpolate(dino_feat[None], size=(new_height, 27), mode='bilinear')[0]
                dino_feat_padded[:, (27 - new_height) // 2:(27 - new_height) // 2 + new_height, :] = dino_feat
            else:
                new_width = int(width / height * 27)
                dino_feat = F.interpolate(dino_feat[None], size=(27, new_width), mode='bilinear')[0]
                dino_feat_padded[:, :, (27 - new_width) // 2:(27 - new_width) // 2 + new_width] = dino_feat
            dino_feat_padded = dino_feat_padded.flatten(1).permute(1, 0)
            merged_image_features.append(torch.cat([clip_feat, dino_feat_padded], dim=-1))
        
        return merged_image_features

    @property
    def dtype(self):
        return self.vision_tower.query_embedding.weight.dtype

    @property
    def device(self):
        return self.vision_tower.query_embedding.weight.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.cfg
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.clip_vision_tower.config.hidden_size

