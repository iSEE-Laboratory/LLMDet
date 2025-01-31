#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo $CHUNKS

CKPT="llava-v1.5-13b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/gqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path checkpoints/my-llava-onevision-qwen2-0.5b-ov-mixed-3/ \
        --question-file /mnt/Datasets/LLaVA/eval/gqa/$SPLIT.jsonl \
        --image-folder /mnt/Datasets/LLaVA/eval/gqa/images \
        --answers-file ./playground/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode qwen_1_5 &
done

wait

output_file=./playground/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python 1_eval.py --tier testdev_balanced --questions /mnt/Datasets/GQA/questions/testdev_balanced_questions.json | tee my_llava_7b_lora_gqa_anyres_acc.txt
