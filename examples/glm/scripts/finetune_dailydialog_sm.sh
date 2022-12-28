#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=/home/tsm/.sat_models
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

echo "main dir $main_dir"
source $main_dir/config/model_glm_large.sh

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
       --train-iters 21 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --valid-data ${eval_data} \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --fp16 \
       --save-interval 21 \
       --eval-interval 10 \
       --save ~/checkpoints \
       --split 1 \
       --strict-eval \
       --eval-batch-size 1 \
       --load ~/checkpoints/finetune-glm-12-27-23-56
"
       # --split 949,50,1 \
       # --load /root/checkpoints/pretrain-bert-mid11-28-15-38 \


gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes 1 --num_gpus 1 --hostfile ${HOST_FILE_PATH} finetune_glm_causal.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
