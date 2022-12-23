MODEL_TYPE="blocklm-large"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 513 \
            --tokenizer-type glm_GPT2BPETokenizer \
            --tokenizer-model-type glm-large \
            --load ${CHECKPOINT_PATH}/glm-large-en-blank"