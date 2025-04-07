import argparse
import json

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import os
from transformers import GroundingDinoProcessor
from modeling_grounding_dino import GroundingDinoForObjectDetection
import tqdm
import random


def build_captions_and_token_span(cat_list, force_lowercase):
    """
    Return:
        captions: str
        cat2tokenspan: dict
            {
                'dog': [[0, 2]],
                ...
            }
    """

    cat2tokenspan = {}
    captions = ""
    for catname in cat_list:
        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()
        if "/" in class_name:
            class_name_list= class_name.strip().split("/")
            class_name_list.append(class_name)
            class_name: str = random.choice(class_name_list)

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(captions) > 0:
                captions = captions + " "
            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:
            captions = captions + " ."
            cat2tokenspan[class_name] = tokens_positive_i

    return captions, cat2tokenspan



def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def main(args):

        
    # 配置参数
    ANN_FILE = args.anno_path
    IMG_DIR = args.image_dir
    MODEL_PATH = args.checkpoint_path
    TEXT_PROMPT = "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush"  # COCO类别提示词

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = GroundingDinoProcessor.from_pretrained(MODEL_PATH)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_PATH).to(device)

    # 加载COCO标注
    coco_gt = COCO(ANN_FILE)
    cat_ids = coco_gt.getCatIds()
    img_ids = coco_gt.getImgIds()

    cat_list = TEXT_PROMPT.split(" . ")
    captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
    tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
    positive_map = create_positive_map_from_span(
        processor.tokenizer(captions), tokenspanlist)  # 80, 256. normed
    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}


    # new_pos_map = torch.zeros((91, 256))
    # for k, v in id_map.items():
    #     new_pos_map[v] = positive_map[k]

    # 结果存储容器
    results = []

    # 推理循环
    for img_id in tqdm.tqdm(img_ids):  # 测试前100张
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(IMG_DIR, img_info['file_name'])
        
        # 预处理
        image = Image.open(img_path).convert("RGB")
        inputs = processor(
            images=image,
            text=TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)

        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = torch.sigmoid(outputs.logits[0])  # (num_boxes, num_classes)
        pos_maps = positive_map.to(logits.device)
        # (bs, 100, 256) @ (80, 256).T -> (bs, 100, 80)
        logits = logits @ pos_maps.T
        boxes = outputs.pred_boxes[0]  # (num_boxes, 4)

        topk_values, topk_indexes = torch.topk(
            logits.view(-1), args.num_select, dim=0)
        logits_filt = topk_values
        topk_boxes = topk_indexes // logits.shape[1]
        labels = topk_indexes % logits.shape[1]
        boxes = box_cxcywh_to_xyxy(boxes)
        boxes_filt = boxes[topk_boxes]
        
        # # 过滤预测结果
        # filt_mask = logits.max(dim=1)[0] > BOX_THRESHOLD
        # logits_filt = logits[filt_mask]
        # boxes_filt = boxes[filt_mask]
        # print(boxes_filt, logits_filt, labels)
        
        # 转换为COCO结果格式
        for box, score, label in zip(boxes_filt, logits_filt, labels):
            xmin, ymin, xmax, ymax = box.cpu().numpy() * torch.tensor(image.size * 2).numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            results.append({
                "image_id": img_id,
                "category_id": id_map[label.item()],
                "bbox": [xmin, ymin, w, h],
                "score": score.max().item()
            })
    # with open("results.json", "w") as f:
    #     json.dump(results, f)

    # 评估指标计算
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on COCO", add_help=True)
    # load model
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # coco info
    parser.add_argument("--anno_path", type=str,
                        required=True, help="coco root")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    args = parser.parse_args()

    main(args)
