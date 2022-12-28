prefix_len=50
max_seq_len=1025
sample_len="$((max_seq_len-prefix_len-1))"
echo $sample_len
MODEL_TYPE="blocklm-10B"

MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --prefix_len $prefix_len \
            --sample_len $sample_len \
            --max-sequence-length $max_seq_len \
            --num-layers 48 \
            --hidden-size 4096 \
            --vocab-size 50304 \
            --num-attention-heads 64 \
            --tokenizer-model-type gpt2 \
            --tokenizer-type glm_GPT2BPETokenizer \
            --load ${CHECKPOINT_PATH}/glm-10b-en"