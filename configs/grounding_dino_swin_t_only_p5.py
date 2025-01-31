_base_ = 'grounding_dino_swin_t.py'

model = dict(
    lmm_max_token_length=1200,
    use_p4_input=False,
)



