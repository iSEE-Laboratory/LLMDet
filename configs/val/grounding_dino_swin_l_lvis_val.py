_base_ = '../grounding_dino_swin_l.py'

model = dict(
    lmm=None,
)

dataset_type = 'LVISV1Dataset'
data_root = '../grounding_data/coco/'

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(1000, 1560),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]


val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file='annotations/lvis_od_val.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline, ))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root + 'annotations/lvis_od_val.json')
test_evaluator = val_evaluator





