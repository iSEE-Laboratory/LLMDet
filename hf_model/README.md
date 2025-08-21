# Use LLMDet in Huggingface🤗

### 1. For users with transformers >= 4.55.0

LLMDet has been merged into `transformers` since `v4.55.0`. We recommend installing the latest version of `transformers`. Therefore, you do not need to change any files.

**Note:** For users using the models in official `transformers`, you should download models at [iSEE-Laboratory](https://huggingface.co/collections/iSEE-Laboratory/llmdet-688475906dc235d5f1dc678e).

____________________________________________________________________________

### 2. For users with transformers < 4.55.0

For users with lower version of `transformers`, you should modify the `modeling_grounding_dino.py` file as below.

Checkpoint: [llmdet_swin_tiny_hf](https://huggingface.co/fushh7/llmdet_swin_tiny_hf), [llmdet_swin_base_hf](https://huggingface.co/fushh7/llmdet_swin_base_hf), [llmdet_swin_large_hf](https://huggingface.co/fushh7/llmdet_swin_large_hf)

【Note】: These checkpoints are not compatible with the one above.

- We first convert mmdet ckpt to GroundingDino ckpt and further convert it to huggingface ckpt. Please refer to `mmdet2groundingdino_swint.py` and `convert_grounding_dino_to_hf.py` for more details. Many thanks to [Tianming Liang](https://github.com/tmliang) for providing the conversion scripts.

- Since LLMDet is similar to GroundingDino, we reuse the code of GroundingDino in Huggingface, but with slightly modifications in `modeling_grounding_dino.py`:

  1. We replace the `GroundingDinoContrastiveEmbedding` in Line 1504-1550.
  2. We fix a shallow copy bug in Line 2995-3002, making it a deep copy.
  3. We change the path in Line 76.

  To load the LLMDet correctly, uses should initialize the model from our provided `modeling_grounding_dino.py`. Other usages are the same as GroundingDino in Huggingface.

- We use `transformers==4.42.0`. Since we find the code in Huggingface varies across different versions. Users with other version should modify the `modeling_grounding_dino.py` accordingly.

- The code in Huggingface has not been thoroughly tested. If encountering any problems, feel free to open an issue.


1. demo

   ```
   python demo_hf.py
   ```

2. Test mAP on COCO val

   ```
   python test_ap_on_coco.py --checkpoint_path llmdet_swin_tiny --anno_path /mnt/data1/yanjunkai/2D/dataset/coco/annotations/instances_val2017.json --image_dir /mnt/data1/yanjunkai/2D/dataset/coco/val2017
   ```
   
   - The results of our tiny hf model on COCO is 54.9, which is slightly lower than the one in mmdet (55.5). But I have no idea where the problem happens☹️. We find the hugginggface version of GroundingDino achieves 47.9 also lower than the one (48.5) in original repo.
