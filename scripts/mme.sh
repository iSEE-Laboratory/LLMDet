#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo $CHUNKS

mmeDIR="./playground/mme"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path checkpoints/my-llava-onevision-qwen2-0.5b-ov-mixed-3/ \
        --question-file /mnt/Datasets/LLaVA/eval/MME_Benchmark_release_version/llava_mme.jsonl \
        --image-folder /mnt/Datasets/LLaVA/eval/MME_Benchmark_release_version/ \
        --answers-file ./playground/mme/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode qwen_1_5 &
done

wait

output_file=./playground/mme/answers/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/mme/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd ./playground/mme

python ../../llava/eval/convert_answer_to_mme.py --prediction_file answers/merge.jsonl --annotation_file /mnt/Datasets/LLaVA/eval/MME_Benchmark_release_version/llava_mme.jsonl

python /mnt/Datasets/LLaVA/eval/MME_Benchmark_release_version/eval_tool/calculation.py --results_dir ./
