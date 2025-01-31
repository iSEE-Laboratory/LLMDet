_base_ = '../grounding_dino_swin_t.py'  # noqa


model = dict(
    lmm=None,
    test_cfg=dict(max_per_img=300, chunked_size=-1)
)


dataset_type = 'CocoDataset'
data_root = '/mnt/Datasets/ood_coco/'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')


# ---------------------1 cartoon---------------------#

_data_root = data_root + 'cartoon/'
dataset_cartoon = dict(
    type=dataset_type,
    data_root=_data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_cartoon = dict(
    type='CocoMetric',
    ann_file=_data_root + 'annotations/instances_val2017.json',
    metric='bbox')

# ---------------------2 handmake---------------------#

_data_root = data_root + 'handmake/'
dataset_handmake = dict(
    type=dataset_type,
    data_root=_data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_handmake = dict(
    type='CocoMetric',
    ann_file=_data_root + 'annotations/instances_val2017.json',
    metric='bbox')

# ---------------------3 painting---------------------#

_data_root = data_root + 'painting/'
dataset_painting = dict(
    type=dataset_type,
    data_root=_data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_painting = dict(
    type='CocoMetric',
    ann_file=_data_root + 'annotations/instances_val2017.json',
    metric='bbox')

# ---------------------4 sketch---------------------#

_data_root = data_root + 'sketch/'
dataset_sketch = dict(
    type=dataset_type,
    data_root=_data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_sketch = dict(
    type='CocoMetric',
    ann_file=_data_root + 'annotations/instances_val2017.json',
    metric='bbox')

# ---------------------5 tattoo---------------------#

_data_root = data_root + 'tattoo/'
dataset_tattoo = dict(
    type=dataset_type,
    data_root=_data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_tattoo = dict(
    type='CocoMetric',
    ann_file=_data_root + 'annotations/instances_val2017.json',
    metric='bbox')

# ---------------------6 weather---------------------#

_data_root = data_root + 'weather/'
dataset_weather = dict(
    type=dataset_type,
    data_root=_data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_weather = dict(
    type='CocoMetric',
    ann_file=_data_root + 'annotations/instances_val2017.json',
    metric='bbox')

# --------------------- Config---------------------#
dataset_prefixes = [
    'cartoon', 'handmake', 'painting', 'sketch',
    'tattoo', 'weather'
]
datasets = [
    dataset_cartoon, dataset_handmake, dataset_painting,
    dataset_sketch, dataset_tattoo, dataset_weather,
]
metrics = [
    val_evaluator_cartoon, val_evaluator_handmake,
    val_evaluator_painting, val_evaluator_sketch,
    val_evaluator_tattoo, val_evaluator_weather,
]

# -------------------------------------------------#
val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
