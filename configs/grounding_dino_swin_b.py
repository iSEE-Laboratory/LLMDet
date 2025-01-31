_base_ = 'grounding_dino_swin_t.py'

model = dict(
    use_autocast=True,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=None)),
    neck=dict(in_channels=[256, 512, 1024]),
)

load_from = '../huggingface/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth'


