#!/bin/bash
CHECKPOINT_PATH=/home/tsm/.sat_models
MPSIZE=1
MAXSEQLEN=512
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.3
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
# main_dir=$(dirname $script_dir)

# echo "main dir $main_dir"
# source $main_dir/config/model_glm_large.sh

python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT inference_glm_small.py \
       --mode inference \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --num-beams 4 \
       --no-repeat-ngram-size 3 \
       --length-penalty 0.7 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --output-path samples_glm \
       --batch-size 1 \
       --out-seq-length 200 \
       --sampling-strategy BeamSearchStrategy
