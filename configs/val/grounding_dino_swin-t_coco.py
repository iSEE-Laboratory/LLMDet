_base_ = '../grounding_dino_swin_t.py'  # noqa


model = dict(
    lmm=None,
    test_cfg=dict(max_per_img=300, chunked_size=-1)
)


dataset_type = 'CocoDataset'
data_root = '/mnt/Datasets/COCO2017/'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')


val_dataloader = dict(dataset=dataset)
test_dataloader = val_dataloader

test_evaluator = val_evaluator
