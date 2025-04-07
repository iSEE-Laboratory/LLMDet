import requests

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import GroundingDinoProcessor
from modeling_grounding_dino import GroundingDinoForObjectDetection


def draw_boxes(image, boxes, labels):
    """
    在图像上绘制边界框和标签，并保存图像。

    :param image: PIL Image对象
    :param boxes: Tensor或列表，格式为 [[x_min, y_min, x_max, y_max], ...]
    :param labels: 标签列表，格式为 [label1, label2, ...]
    """
    # 将图像转化为可编辑
    draw = ImageDraw.Draw(image)
    # 选择文字字体和大小
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        # 绘制边界框
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        # 绘制标签
        draw.text((x_min, y_min - 10), label, fill="red", font=font)

    # 保存图像
    image.save("output_image_with_boxes.jpg")


model_id = "fushh7/llmdet_swin_tiny_hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = GroundingDinoProcessor.from_pretrained(model_id)
model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "a cat. a remote control."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)
print(results)

draw_boxes(image, results[0]['boxes'].cpu().numpy(), results[0]['labels'])
