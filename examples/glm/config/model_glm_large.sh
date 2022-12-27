prefix_len=10
max_seq_len=512
sample_len="$((max_seq_len-prefix_len-1))"
MODEL_TYPE="blocklm-large"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --prefix_len $prefix_len \
            --sample_len $sample_len \
            --max-sequence-length $max_seq_len \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --tokenizer-type glm_GPT2BPETokenizer \
            --tokenizer-model-type gpt2 \
            --load ${CHECKPOINT_PATH}/glm-large-en-blank"