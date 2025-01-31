# from Kosmos-2

from tqdm import tqdm

import torch
import torch.utils.data
import json

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)



class RefExpEvaluatorFromTxt(object):
    def __init__(self, refexp_gt_path, k=(1, -1), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        self.refexp_gt = [json.loads(q) for q in open(refexp_gt_path, "r")]
        print(f"Load {len(self.refexp_gt)} annotations")
        self.k = k
        self.thresh_iou = thresh_iou

    def summarize(self,
                  prediction_file: str,
                  quantized_size: int = 32,
                  verbose: bool = False,):
        
        # get the predictions
        predict_all_lines = [json.loads(q) for q in open(prediction_file, "r")]
        assert len(predict_all_lines) == len(self.refexp_gt)
    
        
        dataset2score = {k: 0.0 for k in self.k}
        dataset2count = 0.0
        for item_img, item_ann in tqdm(zip(predict_all_lines, self.refexp_gt)):
            if item_img['question_id'] != item_ann['question_id']:
                raise ValueError(f"Ann\n{item_ann} \nis not matched\n {item_img}")

            img_height = item_ann['height']
            img_width = item_ann['width']
            target_bbox = item_ann["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            target_bbox = torch.as_tensor(converted_bbox).view(-1, 4)
            
            
            prediction_bbox = item_img['text']
            prediction_bbox = prediction_bbox.split('[')
            if len(prediction_bbox) < 2:
                dataset2count += 1.0
                continue
            prediction_bbox = prediction_bbox[1]
            prediction_bbox = prediction_bbox.split(']')[0]
            prediction_bbox = prediction_bbox.split(',')
            if len(prediction_bbox) == 4:
                prediction_bbox = [float(x.strip()) * factor for x, factor in zip(prediction_bbox, [img_width, img_height, img_width, img_height])]
                predict_boxes = torch.as_tensor(prediction_bbox).view(-1, 4)
            
                iou, _ = box_iou(predict_boxes, target_bbox)
                mean_iou, _ = box_iou(predict_boxes.mean(0).view(-1, 4), target_bbox)
                for k in self.k:
                    if k == 'upper bound':
                        if max(iou) >= self.thresh_iou:
                            dataset2score[k] += 1.0
                    elif k == 'mean':
                        if max(mean_iou) >= self.thresh_iou:
                            dataset2score[k] += 1.0
                    else:
                        if max(iou[0, :k]) >= self.thresh_iou:
                            dataset2score[k] += 1.0

            dataset2count += 1.0

        for k in self.k:
            try:
                dataset2score[k] /= dataset2count
                print(f"Precision @ {k}: {dataset2score[k]} \n")
            except:
                pass
                
        return dataset2score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='prediction_file')
    parser.add_argument('--annotation_file', default='/path/to/mdetr_processed_json_annotations', help='annotation_file')
    parser.add_argument('--quantized_size', default=32, type=int)
    
    args = parser.parse_args()
    
    evaluator = RefExpEvaluatorFromTxt(
        refexp_gt_path=args.annotation_file, 
        k=(1, 'mean', 'upper bound'), 
        thresh_iou=0.5,
    )
    
    evaluator.summarize(args.prediction_file, args.quantized_size, verbose=False)
