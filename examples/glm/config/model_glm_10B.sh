MODEL_TYPE="blocklm-10B"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --prefix_len 200 \
            --num-layers 48 \
            --hidden-size 4096 \
            --vocab-size 50304 \
            --num-attention-heads 64 \
            --max-sequence-length 1025 \
            --tokenizer-model-type gpt2 \
            --tokenizer-type glm_GPT2BPETokenizer \
            --load ${CHECKPOINT_PATH}/glm-10b-en"