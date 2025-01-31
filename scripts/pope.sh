#!/bin/bash


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo $CHUNKS

mmeDIR="./playground/pope"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path checkpoints/my-llava-onevision-qwen2-0.5b-ov-mixed-3/ \
        --question-file /mnt/Datasets/LLaVA/eval/pope/llava_pope.jsonl \
        --image-folder /mnt/Datasets/COCO2014/val2014 \
        --answers-file ./playground/pope/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode qwen_1_5 &
done

wait

output_file=./playground/pope/answers.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/pope/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_pope.py \
    --annotation-dir /mnt/Datasets/LLaVA/eval/pope \
    --question-file /mnt/Datasets/LLaVA/eval/pope/llava_pope.jsonl \
    --result-file ./playground/pope/answers.jsonl
