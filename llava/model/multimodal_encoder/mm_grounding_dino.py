import torch
import torch.nn as nn
from mmdet.utils import register_all_modules
register_all_modules()
from mmengine.config import Config
from mmdet.registry import MODELS
from mmengine.structures import BaseDataElement
from mmcv.ops.nms import nms
from mmdet.datasets.transforms import FixScaleResize, PackDetInputs
import numpy as np
import PIL.Image as Image

from ram.models import ram_plus
from ram import get_transform

class GroundingDINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.grounding_dino_config = args.grounding_dino_config
        self.vision_tower_weight_path = args.vision_tower_weight_path.split('+')
        self.load_ram = args.load_ram
        self.bert_base_path = args.bert_base_path
        # self.vision_tower_weight_path = args.vision_tower_weight_path
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

    def load_model(self, device_map='cuda'):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
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

    @torch.no_grad()
    def forward(self, images, tags):
        self.vision_tower.eval()
        if type(images) is Image.Image:
            images = [images]
        
        if tags is None or tags[0] is None:
            ram_input_image = [self.ram_transform(img) for img in images]
            ram_input_image = torch.stack(ram_input_image, dim=0)
            ram_input_image = ram_input_image.to(self.device).to(self.dtype)
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

        image_features = self.vision_tower.predict_encoder_only(batched_results['inputs'].to(self.dtype), data_samples)
        # output = self.vision_tower.predict(batched_results['inputs'].to(self.dtype), data_samples)

        # image_features = []
        # for out, img_shape in zip(output, img_shapes):
        #     scores = out.pred_instances.scores
        #     boxes = out.pred_instances.bboxes
        #     bbox_index = out.pred_instances.bbox_index
        #     boxes, nms_inds = nms(boxes.float(), scores.float(), iou_threshold=0.3)
        #     scores = boxes[:, 4].to(self.dtype)
        #     boxes = boxes[:, :4].to(self.dtype)
        #     bbox_index = bbox_index[nms_inds]
        #     normalized_boxes = boxes / torch.tensor([img_shape[1], img_shape[0], img_shape[1], img_shape[0]], device=boxes.device, dtype=boxes.dtype)

        #     w = normalized_boxes[:, 2] - normalized_boxes[:, 0]
        #     h = normalized_boxes[:, 3] - normalized_boxes[:, 1]
        #     normalized_boxes = normalized_boxes[(w > 0.01) & (h > 0.01)]
        #     bbox_index = bbox_index[(w > 0.01) & (h > 0.01)]

        #     features = out.object_query[bbox_index]
        #     image_features.append((features, normalized_boxes))
        
        # for image, (features, normalized_boxes), img_shape in zip(images, image_features, img_shapes):
        #     boxes = normalized_boxes * torch.tensor([img_shape[1], img_shape[0], img_shape[1], img_shape[0]], device=boxes.device, dtype=boxes.dtype)
        #     import torchvision
        #     image = torchvision.transforms.ToTensor()(image) * 255
        #     image = image.to(torch.uint8)
        #     img = torchvision.utils.draw_bounding_boxes(image, boxes.float(), colors='blue', width=6)
        #     img = img.permute(1, 2, 0).cpu().numpy()
        #     img = Image.fromarray(img)
        #     img.save('%d.jpg' % (self.img_idx))
        #     self.img_idx += 1
        
        return image_features

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
        return 256

