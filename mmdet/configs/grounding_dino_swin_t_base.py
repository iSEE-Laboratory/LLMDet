from mmcv.transforms import RandomChoiceResize
from mmcv.transforms.loading import LoadImageFromFile
from mmengine.model.weight_init import PretrainedInit
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
from mmengine.dataset.sampler import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.runner import LogProcessor
from mmengine.visualization import LocalVisBackend

from mmdet.engine.hooks import GroundingVisualizationHook
from mmdet.visualization import DetLocalVisualizer
from torch.nn.modules.normalization import GroupNorm
from torch.optim.adamw import AdamW

from mmdet.datasets import (ODVGDataset, LVISV1Dataset, ConcatDataset)
from mmdet.datasets.samplers import AspectRatioBatchSampler
from mmdet.datasets.transforms import (LoadAnnotations, Resize, PackDetInputs, RandomFlip, RandomSamplingNegPos2, FilterAnnotations, FixScaleResize)
from mmdet.models import (GroundingDINO, ChannelMapper, DetDataPreprocessor, BertModel,
                          SwinTransformer, GroundingDINOHead)
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.models.task_modules import (BBoxL1Cost, BinaryFocalLossCost,
                                       HungarianAssigner, IoUCost)
from mmdet.evaluation.metrics.lvis_metric import LVISFixedAPMetric

pretrained = None  # noqa
lang_model_name = '../../huggingface/bert-base-uncased/'

model = dict(
    type=GroundingDINO,
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    lmm='../../huggingface/Qwen2-0.5B-Instruct-llava/',
    lmm_layers=1,
    lmm_loss_weight=1.0,
    lmm_connector='../../huggingface/mm_projector.bin',
    fsdp=True,
    freeze_backbone=True,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type=BertModel,
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type=SwinTransformer,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type=PretrainedInit, checkpoint=pretrained)),
    neck=dict(
        type=ChannelMapper,
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type=GroupNorm, num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type=GroundingDINOHead,
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type=L1Loss, loss_weight=5.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=BinaryFocalLossCost, weight=2.0),
                dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                dict(type=IoUCost, iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300, chunked_size=40,))




# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=RandomFlip, prob=0.5),
    # dict(
    #     type='RandomChoice',
    #     transforms=[
    #         [
    #             dict(
    #                 type='RandomChoiceResize',
    #                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                         (736, 1333), (768, 1333), (800, 1333)],
    #                 keep_ratio=True)
    #         ],
    #         [
    #             dict(
    #                 type='RandomChoiceResize',
    #                 # The radio of all image in train dataset < 7
    #                 # follow the original implement
    #                 scales=[(400, 4200), (500, 4200), (600, 4200)],
    #                 keep_ratio=True),
    #             dict(
    #                 type='RandomCrop',
    #                 crop_type='absolute_range',
    #                 crop_size=(384, 600),
    #                 allow_negative_crop=True),
    #             dict(
    #                 type='RandomChoiceResize',
    #                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                         (736, 1333), (768, 1333), (800, 1333)],
    #                 keep_ratio=True)
    #         ]
    #     ]),
    dict(
        type=RandomChoiceResize,
        resize_type=Resize,
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type=RandomSamplingNegPos2,
        tokenizer_name=lang_model_name,
        tokenizer_name2='../../huggingface/Qwen2-0.5B-Instruct-llava/',
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode', 'conversations'))
]

test_pipeline = [
    dict(
        type=LoadImageFromFile, backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type=FixScaleResize,
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]


# --------------------------- coco2017 od dataset---------------------------
coco2017_train_dataset = dict(
    type=ODVGDataset,
    data_root='../../grounding_data/coco/',
    ann_file='annotations/instances_train2017_od_merged2.json',
    label_map_file='annotations/coco2017_label_map.json',
    data_prefix=dict(img='train2017'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- flickr30k vg dataset---------------------------
flickr30k_dataset = dict(
    type=ODVGDataset,
    data_root='../../grounding_data/flickr30k_entities/',
    ann_file='flickr_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- gqa vg dataset---------------------------
gqa_dataset = dict(
    type=ODVGDataset,
    data_root='../../grounding_data/gqa/',
    ann_file='gqa_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- gqa vg dataset---------------------------
caption_dataset = dict(
    type=ODVGDataset,
    data_root='../../grounding_data/llava_cap/',
    ann_file='LLaVA-ReCap-558K_tag_box_vg.json',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(type=ConcatDataset, datasets=[
        coco2017_train_dataset,
        flickr30k_dataset,
        gqa_dataset,
        caption_dataset,
    ]))


val_dataloader = dict(
    dataset=dict(
        data_root='../../grounding_data/coco/',
        type=LVISV1Dataset,
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline, 
        return_classes=True))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    type=LVISFixedAPMetric,
    ann_file='../../grounding_data/coco/' +
    'annotations/lvis_v1_minival_inserted_image_name.json')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=0.0001,
                   weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))


max_iter = 180000
train_cfg = dict(
    type=IterBasedTrainLoop,
    max_iters=max_iter,
    val_interval=30000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[120000, 150000],
        gamma=0.1)
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    sampler_seed=dict(type=DistSamplerSeedHook),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=30000, max_keep_ckpts=30),
    visualization=dict(type=GroundingVisualizationHook),
    logger=dict(type=LoggerHook, interval=100))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=DetLocalVisualizer, vis_backends=vis_backends, name='visualizer')
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

# # learning policy
# max_epochs = 4
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[2, 3],
#         gamma=0.1)
# ]

# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enable=True)

# default_hooks = dict(visualization=dict(type='GroundingVisualizationHook'))

load_from = '../../huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
log_level = 'INFO'
resume = False
