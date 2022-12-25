#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=/home/tsm/.sat_models/glm-10b-en
MODEL_TYPE="blocklm-10B"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 48 \
            --hidden-size 4096 \
            --num-attention-heads 64 \
            --max-sequence-length 1025 \
            --tokenizer-model-type gpt2 \
            --tokenizer-type glm_GPT2BPETokenizer \
            --load ${CHECKPOINT_PATH}/glm-10b-en"
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

echo "main dir $main_dir"
source $main_dir/config/model_glm_10B.sh

echo "$MODEL_ARGS"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

en_data="hf://daily_dialog/default/train"
eval_data="daily_dialog/default/validation"
test_data="daily_dialog/default/test"

config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name finetune-glm \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 6000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --valid-data ${eval_data} \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --fp16 \
       --save-interval 6000 \
       --eval-interval 100 \
       --save /root/checkpoints \
       --split 1 \
       --strict-eval \
       --eval-batch-size 1
"
       # --load  /root/checkpoints/pretrain-bert-mid-std-fulltrain12-02-06-10
       #  \       --sandwich-ln
       # --split 949,50,1 \
       # --load /root/checkpoints/pretrain-bert-mid11-28-15-38 \


gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes 1 --num_gpus 1 --hostfile ${HOST_FILE_PATH} finetune_regressive.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
