# Use LLMDet in Huggingface🤗

Checkpoint:

[llmdet_swin_tiny_hf](https://huggingface.co/fushh7/llmdet_swin_tiny_hf), [llmdet_swin_base_hf](https://huggingface.co/fushh7/llmdet_swin_base_hf), [llmdet_swin_large_hf](https://huggingface.co/fushh7/llmdet_swin_large_hf)

1. demo

   ```
   python demo_hf.py
   ```

2. Test mAP on COCO val

   ```
   python test_ap_on_coco.py --checkpoint_path llmdet_swin_tiny --anno_path /mnt/data1/yanjunkai/2D/dataset/coco/annotations/instances_val2017.json --image_dir /mnt/data1/yanjunkai/2D/dataset/coco/val2017
   ```
   
   - The results of our tiny hf model on COCO is 54.9, which is slightly lower than the one in mmdet (55.5). But I have no idea where the problem happens☹️. We find the hugginggface version of GroundingDino achieves 47.9 also lower than the one (48.5) in original repo.

Note:

- We first convert mmdet ckpt to GroundingDino ckpt and further convert it to huggingface ckpt. Please refer to `mmdet2groundingdino_swint.py` and `convert_grounding_dino_to_hf.py` for more details. Many thanks to [Tianming Liang](https://github.com/tmliang) for providing the conversion scripts.

- Since LLMDet is similar to GroundingDino, we reuse the code of GroundingDino in Huggingface, but with slightly modifications in `modeling_grounding_dino.py`:

  1. We replace the `GroundingDinoContrastiveEmbedding` in Line 1504-1550.
  2. We fix a shallow copy bug in Line 2995-3002, making it a deep copy.
  3. We change the path in Line 76.

  To load the LLMDet correctly, uses should initialize the model from our provided `modeling_grounding_dino.py`. Other usages are the same as GroundingDino in Huggingface.

- We use `transformers==4.42.0`. Since we find the code in Huggingface varies across different versions. Users with other version should modify the `modeling_grounding_dino.py` accordingly.

- The code in Huggingface has not been thoroughly tested. If encountering any problems, feel free to open an issue.
