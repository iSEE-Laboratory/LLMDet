_base_ = [
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py', '_base_/default_runtime.py'
]

pretrained = None  # noqa
lang_model_name = '../huggingface/bert-base-uncased/'
lmm_path = '../huggingface/my_llava-onevision-qwen2-0.5b-ov-2/'
lmm_max_token_length = 1600
num_region_caption = 16
use_short_cap=False
use_uniform_prompt=True
clean_caption=True

randomness=dict(seed=624982218)

model = dict(
    type='GroundingDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    lmm=lmm_path,
    lmm_max_token_length=lmm_max_token_length,
    lmm_region_loss_weight=1.0,
    lmm_image_loss_weight=1.0,
    lmm_connector='../huggingface/my_llava-onevision-qwen2-0.5b-ov-2/mm_projector2.bin',
    lmm_connector_prefix='mm_projector',
    freeze_backbone=True,
    freeze_lm=False,
    use_plora=False,
    use_lora=True,
    use_lmm_cross_attn=True,
    num_lmm_new_layers=6,
    lmm_new_layer_insert_type='all',
    feature_map_size=27,
    num_region_caption=num_region_caption,
    use_p5_input=True,
    use_p4_input=True,
    use_query_input=False,
    use_image_level_cross_attn=False,
    mini_query=False,
    lora_r=128, 
    lora_alpha=256, 
    lora_dropout=0,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
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
        type='GroundingDINOHead',
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300, chunked_size=40,))




# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos2',
        tokenizer_name=lang_model_name,
        tokenizer_name2=lmm_path,
        lmm_max_token_length=lmm_max_token_length,
        num_region_caption=num_region_caption,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text', 'tags', 'contrast_conv',
                   'custom_entities', 'tokens_positive', 'dataset_mode', 'conversations', 'region_conversations'))
]


test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]


# --------------------------- coco2017 od dataset---------------------------
coco2017_train_dataset = dict(
    type='ODVGDataset',
    data_root='../grounding_data/coco/',
    ann_file='annotations/instances_train2017_vg_merged6.jsonl',
    data_prefix=dict(img='train2017'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    actual_dataset_mode='OD',
    use_short_cap=use_short_cap,
    use_uniform_prompt=use_uniform_prompt,
    clean_caption=clean_caption,
    backend_args=None)

# --------------------------- flickr30k vg dataset---------------------------
flickr30k_dataset = dict(
    type='ODVGDataset',
    data_root='../grounding_data/flickr30k_entities/',
    ann_file='flickr_train_vg7.jsonl',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    actual_dataset_mode='VG',
    use_short_cap=use_short_cap,
    use_uniform_prompt=use_uniform_prompt,
    clean_caption=clean_caption,
    backend_args=None)

# --------------------------- gqa vg dataset---------------------------
gqa_dataset = dict(
    type='ODVGDataset',
    data_root='../grounding_data/gqa/',
    ann_file='gqa_train_vg7.jsonl',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    actual_dataset_mode='VG',
    use_short_cap=use_short_cap,
    use_uniform_prompt=use_uniform_prompt,
    clean_caption=clean_caption,
    backend_args=None)

# --------------------------- gqa vg dataset---------------------------
caption_dataset = dict(
    type='ODVGDataset',
    data_root='../grounding_data/llava_cap/',
    ann_file='LLaVA-ReCap-558K_tag_box_vg7.jsonl',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    actual_dataset_mode='OD',
    use_short_cap=use_short_cap,
    use_uniform_prompt=use_uniform_prompt,
    clean_caption=clean_caption,
    backend_args=None)

# --------------------------- v3det vg dataset---------------------------
v3det_dataset = dict(
    type='ODVGDataset',
    data_root='../grounding_data/v3det/',
    ann_file='annotations/v3det_2023_v1_train_vg7.jsonl',
    label_map_file=None,
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    actual_dataset_mode='OD',
    use_short_cap=use_short_cap,
    use_uniform_prompt=use_uniform_prompt,
    clean_caption=clean_caption,
    backend_args=None)

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[
        coco2017_train_dataset,
        flickr30k_dataset,
        gqa_dataset,
        caption_dataset,
        v3det_dataset,
    ]))

dataset_type = 'LVISV1Dataset'
data_root = '../grounding_data/coco/'

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline, 
        return_classes=True))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_minival_inserted_image_name.json')
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001,
                   weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
            # 'lmm': dict(lr_mult=0.1),
        }))


max_iter = 150000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=30000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[120000, 140000],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=30000, max_keep_ckpts=30),
    visualization=dict(type='GroundingVisualizationHook'),
    logger=dict(type='LoggerHook', interval=100))
log_processor = dict(by_epoch=False)

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

env_cfg = dict(
    dist_cfg=dict(backend='nccl', timeout=36000), # 36000s = 10h
)

load_from = '../huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'

