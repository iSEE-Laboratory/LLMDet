# mmdet to groundingdino
import argparse
from collections import OrderedDict
import torch
from mmengine.runner import CheckpointLoader

# convert the functions from mmdet to groundingdino
def correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, in_channel // 4, 4).transpose(1, 2)
    x = x[:, [0, 2, 1, 3], :]
    x = x.reshape(out_channel, in_channel)
    return x

def correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(in_channel // 4, 4).transpose(0, 1)
    x = x[[0, 2, 1, 3], :]
    x = x.reshape(in_channel)
    return x

def convert(ckpt):
    """Inverse mapping of checkpoint parameters to their original names."""
    # Create a dictionary to hold the reversed checkpoint
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v  # Start with the original value

        # Inverse rules based on the convert function (from specific to general)
        if k.startswith('decoder'):
            new_k = k.replace('decoder', 'transformer.decoder')
            if 'norms.2' in new_k:
                new_k = new_k.replace('norms.2', 'norm1')
            if 'norms.1' in new_k:
                new_k = new_k.replace('norms.1', 'catext_norm')
            if 'norms.0' in new_k:
                new_k = new_k.replace('norms.0', 'norm2')
            if 'norms.3' in new_k:
                new_k = new_k.replace('norms.3', 'norm3')
            if 'cross_attn_text' in new_k:
                new_k = new_k.replace('cross_attn_text', 'ca_text')
                new_k = new_k.replace('attn.in_proj_weight', 'in_proj_weight')
                new_k = new_k.replace('attn.in_proj_bias', 'in_proj_bias')
                new_k = new_k.replace('attn.out_proj.weight', 'out_proj.weight')
                new_k = new_k.replace('attn.out_proj.bias', 'out_proj.bias')
            if 'ffn.layers.0.0' in new_k:
                new_k = new_k.replace('ffn.layers.0.0', 'linear1')
            if 'ffn.layers.1' in new_k:
                new_k = new_k.replace('ffn.layers.1', 'linear2')
            if 'self_attn.attn' in new_k:
                new_k = new_k.replace('self_attn.attn', 'self_attn')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        # encoder部分最后的reg_layer_id是6，和decoder区分开来
        elif k.startswith('bbox_head.reg_branches.6'):
            if k.startswith('bbox_head.reg_branches.6.0'):
                new_k = k.replace('bbox_head.reg_branches.6.0',
                                  'transformer.enc_out_bbox_embed.layers.0')
            if k.startswith('bbox_head.reg_branches.6.2'):
                new_k = k.replace('bbox_head.reg_branches.6.2',
                                  'transformer.enc_out_bbox_embed.layers.1')
            if k.startswith('bbox_head.reg_branches.6.4'):
                new_k = k.replace('bbox_head.reg_branches.6.4',
                                  'transformer.enc_out_bbox_embed.layers.2')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('query_embedding'):
            new_k = k.replace('query_embedding', 'transformer.tgt_embed')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('bbox_head.reg_branches'):
            # mmdet直接省略了参数名的一部分，需要查看groundingdino的checkpoint
            # groundingdino有两部分参数值是一致的
            # 分别是bbox_embed和transformer.decoder.embed
            # 所以mmdet直接将两部分参数进行了“合并”
            reg_layer_id = int(k.split('.')[2])
            linear_id = int(k.split('.')[3])
            weight_or_bias = k.split('.')[-1]
            new_k1 = 'transformer.decoder.bbox_embed.' + \
                    str(reg_layer_id) + '.layers.' + str(linear_id // 2) + '.' + weight_or_bias
            new_k2 = 'bbox_embed.' + \
                     str(reg_layer_id) + '.layers.' + str(linear_id // 2) + '.' + weight_or_bias

            new_ckpt[new_k1] = new_v  # Add the key and value to the original checkpoint dict
            new_ckpt[new_k2] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('bbox_head.cls_branches.6'):
            # mmdet在contrastive_embed中添加了bias项
            # 但是decoder应该是0~5，所以6应该是采取两阶段微调后对应的enc_out.class_embed
            new_k = 'transformer.enc_out_class_embed.bias'

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('bbox_head.cls_branches'):
            # mmdet在contrastive_embed中添加了bias项
            new_k1 = 'transformer.decoder.class_embed.' + k[-6:]
            new_k2 = 'class_embed.' + k[-6:]

            new_ckpt[new_k1] = new_v  # Add the key and value to the original checkpoint dict
            new_ckpt[new_k2] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('memory_trans_'):
            if k.startswith('memory_trans_fc'):
                new_k = k.replace('memory_trans_fc', 'transformer.enc_output')
            elif k.startswith('memory_trans_norm'):
                new_k = k.replace('memory_trans_norm', 'transformer.enc_output_norm')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('encoder'):
            new_k = k.replace('encoder', 'transformer.encoder')
            new_k = new_k.replace('norms.0', 'norm1')
            new_k = new_k.replace('norms.1', 'norm2')
            new_k = new_k.replace('norms.2', 'norm3')
            new_k = new_k.replace('ffn.layers.0.0', 'linear1')
            new_k = new_k.replace('ffn.layers.1', 'linear2')
            if 'text_layers' in new_k:
                new_k = new_k.replace('self_attn.attn', 'self_attn')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('level_embed'):
            new_k = k.replace('level_embed', 'transformer.level_embed')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('neck.convs'):
            new_k = k.replace('neck.convs', 'input_proj')
            new_k = new_k.replace('neck.extra_convs.0', 'neck.convs.3')
            new_k = new_k.replace('conv.weight', '0.weight')
            new_k = new_k.replace('conv.bias', '0.bias')
            new_k = new_k.replace('gn.weight', '1.weight')
            new_k = new_k.replace('gn.bias', '1.bias')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif 'neck.extra_convs.0' in k:
            new_k = k.replace('neck.extra_convs.0', 'neck.convs.4')
            new_k = new_k.replace('neck.convs', 'input_proj')
            new_k = new_k.replace('conv.weight', '0.weight')
            new_k = new_k.replace('conv.bias', '0.bias')
            new_k = new_k.replace('gn.weight', '1.weight')
            new_k = new_k.replace('gn.bias', '1.bias')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('text_feat_map'):
            new_k = k.replace('text_feat_map', 'feat_map')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('language_model.language_backbone.body.model'):
            new_k = k.replace('language_model.language_backbone.body.model', 'bert')

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        elif k.startswith('backbone'):
            new_k = k.replace('backbone', 'backbone.0')
            if 'patch_embed.projection' in new_k:
                new_k = new_k.replace('patch_embed.projection', 'patch_embed.proj')
            elif 'drop_after_pos' in new_k:
                new_k = new_k.replace('drop_after_pos', 'pos_drop')

            if 'stages' in new_k:
                new_k = new_k.replace('stages', 'layers')
                if 'ffn.layers.0.0' in new_k:
                    new_k = new_k.replace('ffn.layers.0.0', 'mlp.fc1')
                elif 'ffn.layers.1' in new_k:
                    new_k = new_k.replace('ffn.layers.1', 'mlp.fc2')
                elif 'attn.w_msa' in new_k:
                    new_k = new_k.replace('attn.w_msa', 'attn')

                if 'downsample' in k:
                    if 'reduction.' in k:
                        new_v = correct_unfold_reduction_order(v)
                    elif 'norm.' in k:
                        new_v = correct_unfold_norm_order(v)

            new_ckpt[new_k] = new_v  # Add the key and value to the original checkpoint dict

        #########################################################################

        else:
            print('skip:', k)
            continue

        # if 'transformer.decoder.bbox_embed' in new_k:
        #     new_k = new_k.replace('transformer.decoder.bbox_embed', 'bbox_embed')
        # if new_k.startswith('module.'):
        #     new_k = new_k.replace('module.', '')

    return new_ckpt

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys to GroundingDINO style.')
    parser.add_argument(
        'src',
        nargs='?',
        default='grounding_dino_swin-l_pretrain_all-56d69e78.pth',
        help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument(
        'dst',
        nargs='?',
        default='mmdet_swinl.pth_groundingdino.pth',
        help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    # mmdet中是state_dict而不是model
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert(state_dict)
    torch.save(weight, args.dst)
    # sha = subprocess.check_output(['sha256sum', args.dst]).decode()
    # sha = calculate_sha256(args.dst)
    # final_file = args.dst.replace('.pth', '') + '-{}.pth'.format(sha[:8])
    # subprocess.Popen(['mv', args.dst, final_file])
    print(f'Done!!, save to {args.dst}')

if __name__ == '__main__':
    main()

# skip: dn_query_generator.label_embedding.weight