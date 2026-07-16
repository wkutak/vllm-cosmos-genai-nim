#!/bin bash

ngc registry model download-version --org nvidian nvidian/cosmos3:cosmos3-nano-fp8-v2.0-720x1280-50steps-75bfc022-03c14e74 && \
vllm serve /vllm-workspace/cosmos3_vcosmos3-nano-fp8-v2.0-720x1280-50steps-75bfc022-03c14e74 \
    --served-model-name cosmos3-nano-reasoner-fp8 \
    --async-scheduling \
    --allowed-local-media-path "$(pwd)" \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}' \
    --enable-tokenizer-info-endpoint \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_mm_encoder": true}' \
    --port 8080