# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

try:
    from transformers import AutoTokenizer
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import re

import numpy as np
import copy
import torch

def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = name.lower()
    return name


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer,
                                max_tokens):
    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_list = np.unique(gt_labels).tolist()
    # random shuffule so we can sample different annotations
    # at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(text[str(label)]) + '. '

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)
            keep_gt_labels.append(gt_labels[i])

    return gt_bboxes[keep_box_index], np.array(
        keep_gt_labels, dtype=np.long), length


def generate_senetence_given_labels(positive_label_list, negative_label_list,
                                    text):
    label_to_positions = {}

    label_list = negative_label_list + positive_label_list

    random.shuffle(label_list)

    pheso_caption = ''

    label_remap_dict = {}
    for index, label in enumerate(label_list):

        start_index = len(pheso_caption)

        pheso_caption += clean_name(text[str(label)])

        end_index = len(pheso_caption)

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[int(label)] = index

        # if index != len(label_list) - 1:
        #     pheso_caption += '. '
        pheso_caption += '. '

    return label_to_positions, pheso_caption, label_remap_dict


@TRANSFORMS.register_module()
class RandomSamplingNegPos(BaseTransform):

    def __init__(self,
                 tokenizer_name,
                 num_sample_negative=85,
                 max_tokens=256,
                 full_sampling_prob=0.5,
                 label_map_file=None):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.label_map = None
        if label_map_file:
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            return self.od_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        # TODO: add neg
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']

        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'.
                  format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                if int(i) not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = \
            generate_senetence_given_labels(positive_label_list,
                                            negative_label_list, text)

        # label remap
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        return results


def clean_phrase(phrase: str):
    phrase = phrase.lower().strip('.')
    delete_words = ['a ', 'an ', 'the ', 'one ', 'two ', 'three ', 'four ', 'five ', 'six ', 'seven ', 'eigth ', 'nine ']
    for word in delete_words:
        if phrase.startswith(word):
            phrase = phrase[len(word):]
    return phrase.capitalize() + '.'


@TRANSFORMS.register_module()
class RandomSamplingNegPos2(BaseTransform):

    def __init__(self,
                 tokenizer_name,
                 tokenizer_name2,
                 num_sample_negative=85,
                 lmm_max_token_length=512,
                 max_tokens=256,
                 num_region_caption=0,
                 full_sampling_prob=0.5,
                 label_map_file=None):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.lmm_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name2,
                cache_dir=None, 
                model_max_length=lmm_max_token_length, 
                padding_side="right")
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.num_region_caption = num_region_caption
        self.label_map = None
        if label_map_file:
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)
        from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
        self.IGNORE_INDEX = IGNORE_INDEX
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        self.roles = {"human": "user", "gpt": "assistant"}

        # Add image tokens to tokenizer as a special tokens
        # When there is actually an image, we add the image tokens as a special token
        self.lmm_tokenizer.add_tokens(["<image>"], special_tokens=True)

        self.image_token_index = self.lmm_tokenizer.convert_tokens_to_ids("<image>")
        im_start, im_end = self.lmm_tokenizer.additional_special_tokens_ids
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        self.unmask_tokens_idx =  [198, im_start, im_end]

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.lmm_tokenizer.chat_template = chat_template

        self.image_level_input_id = None
        self.image_level_target = None
        self.region_level_input_id = None
        self.region_level_target = None


    def get_LMM_input(self, type, source):

        input_id, target = [], []
        if type == 'image':
            if self.image_level_input_id is not None:
                input_id = copy.deepcopy(self.image_level_input_id)
                target = copy.deepcopy(self.image_level_target)
            else:
                # Build system message for each sentence
                input_id += self.lmm_tokenizer.apply_chat_template([{"role" : "system", "content" : "You are a helpful assistant."}])
                target += [self.IGNORE_INDEX] * len(input_id)

                conv = source[0]
                role = conv["from"]
                content = conv["value"]

                role =  self.roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                encode_id = self.lmm_tokenizer.apply_chat_template(conv)
                input_id += encode_id
                target += [self.IGNORE_INDEX] * len(encode_id)

                for idx, encode_id in enumerate(input_id):
                    if encode_id in self.unmask_tokens_idx:
                        target[idx] = encode_id
                    if encode_id == self.image_token_index:
                        input_id[idx] = self.IMAGE_TOKEN_INDEX
                
                self.image_level_input_id = copy.deepcopy(input_id)
                self.image_level_target = copy.deepcopy(target)

        elif type == 'region':
            if self.region_level_input_id is not None:
                input_id = copy.deepcopy(self.region_level_input_id)
                target = copy.deepcopy(self.region_level_target)
            else:
                # Build system message for each sentence
                input_id += self.lmm_tokenizer.apply_chat_template([{"role" : "system", "content" : "You are a helpful assistant."}])
                target += [self.IGNORE_INDEX] * len(input_id)

                conv = source[0]
                role = conv["from"]
                content = conv["value"]

                role =  self.roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                encode_id = self.lmm_tokenizer.apply_chat_template(conv)
                input_id += encode_id
                target += [self.IGNORE_INDEX] * len(encode_id)

                for idx, encode_id in enumerate(input_id):
                    if encode_id in self.unmask_tokens_idx:
                        target[idx] = encode_id
                    if encode_id == self.image_token_index:
                        input_id[idx] = self.IMAGE_TOKEN_INDEX
                
                self.region_level_input_id = copy.deepcopy(input_id)
                self.region_level_target = copy.deepcopy(target)
        
        else:
            raise NotImplementedError
        
        conv = source[1]
        role = conv["from"]
        content = conv["value"]

        role =  self.roles.get(role, role)
        
        conv = [{"role" : role, "content" : content}]
        encode_ids = self.lmm_tokenizer.apply_chat_template(conv)
        sub_target = copy.deepcopy(encode_ids)

        for idx, encode_id in enumerate(encode_ids):
            if encode_id in self.unmask_tokens_idx:
                sub_target[idx] = encode_id
            if encode_id == self.image_token_index:
                encode_ids[idx] = self.IMAGE_TOKEN_INDEX
        
        input_id += encode_ids
        target += sub_target
        
        return dict(
            input_id=input_id,
            label=target,
        )
        

    def transform(self, results: dict) -> dict:
        ori_conv = copy.deepcopy(results['conversations'])
        results['conversations'] = self.get_LMM_input('image', results['conversations'])
        results['ori_conv'] = ori_conv
        if self.num_region_caption > 0:
            box_index = torch.randperm(len(results['gt_bboxes_labels']))[:self.num_region_caption]
            region_conversations = []
            region_index = results['gt_bboxes_labels'][box_index.numpy()]
            for idx in region_index:
                try:
                    phrase = random.choice(results['phrases'][idx]['phrase']) if type(results['phrases'][idx]['phrase']) is list else results['phrases'][idx]['phrase']
                except:
                    phrase = 'object'
                conv = self.get_LMM_input('region', [{"from": "human", "value": "<image>\nDescribe the region in a phrase."}, {"from": "gpt", "value": clean_phrase(phrase)}])
                region_conversations.append(conv)
            results['region_conversations'] = {'box_index': box_index, 'conversations': region_conversations}
            
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            return self.od_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        # TODO: add neg
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']

        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'.
                  format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                if int(i) not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = \
            generate_senetence_given_labels(positive_label_list,
                                            negative_label_list, text)

        # label remap
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        return results






@TRANSFORMS.register_module()
class LoadTextAnnotations(BaseTransform):

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            tokens_positive = [
                phrase['tokens_positive']
                for phrase in results['phrases'].values()
            ]
            results['tokens_positive'] = tokens_positive
        else:
            text = results['text']
            results['text'] = list(text.values())
        return results
