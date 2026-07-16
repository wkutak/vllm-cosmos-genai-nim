# Cosmos3 quantization perf benchmark.
# Edit the CONFIG block in bench_cosmos3.sh (checkpoint paths) before running.

IMAGE_NAME := vllm-local
IMAGE_NAME_CUDA129 := $(IMAGE_NAME)-cuda129
IMAGE_VERSION := $(shell git rev-parse --short HEAD)

PROJECT_ROOT := $(shell realpath $(shell dirname "$(PWD)"))
SANDBOX_DIR := $(PROJECT_ROOT)/.sandbox

SCRATCH_SPACE ?= /home/scratch.wkutak_other/
DATA_SPACE ?= /home/scratch.wkutak_other_1/
MOUNTS ?= -v $(SCRATCH_SPACE):$(SCRATCH_SPACE) -v $(DATA_SPACE):$(DATA_SPACE) -v $(SANDBOX_DIR):/sandbox

# Serve config (override on the command line, e.g. `make serve-cosmos3 CKPT=... PORT=8000`)
DATA_ROOT      ?= $(DATA_SPACE)dev/cosmos3/quantization/data/cosmos3-nano
CKPT           ?= $(DATA_ROOT)/fp8
SERVE_PORT     ?= 8123
TP_SIZE        ?= 1
GPU_MEM_UTIL   ?= 0.9
LIMIT_MM       ?= {"image": 1}
VLLM_BIN       ?= vllm
EXTRA          ?=

.PHONY: bench-cosmos3 bench-cosmos3-clean serve-cosmos3


build:
	DOCKER_BUILDKIT=1 docker build \
		-t $(IMAGE_NAME):$(IMAGE_VERSION) \
		-f Dockerfile.local \
		$(PROJECT_ROOT)
	@echo Image $(IMAGE_NAME) built.

build-reasoner:
	DOCKER_BUILDKIT=1 docker build \
		-t $(IMAGE_NAME)-reasoner:$(IMAGE_VERSION) \
		-f Dockerfile.local.reasoner \
		$(PROJECT_ROOT)
	@echo Image $(IMAGE_NAME) built.

run:	
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-p $(HOST_PORT):8000 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		--entrypoint bash \
		$(IMAGE_NAME):$(IMAGE_VERSION)

run-reasoner:	
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-p $(HOST_PORT):8000 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		--entrypoint bash \
		$(IMAGE_NAME)-reasoner:$(IMAGE_VERSION)
	
upload:
	docker tag $(IMAGE_NAME):$(IMAGE_VERSION) nvcr.io/nvidian/vllm-cosmos3:v0.25.0_git_vllm_$(IMAGE_VERSION) && \
	docker push nvcr.io/nvidian/vllm-cosmos3:v0.25.0_git_vllm_$(IMAGE_VERSION)

upload-reasoner:
	docker tag $(IMAGE_NAME)-reasoner:$(IMAGE_VERSION) nvcr.io/nvidian/vllm-cosmos3:v0.25.0_reasoner-overlay_git_vllm_$(IMAGE_VERSION) && \
	docker push nvcr.io/nvidian/vllm-cosmos3:v0.25.0_reasoner-overlay_git_vllm_$(IMAGE_VERSION)


# Run the full 3-checkpoint benchmark and emit cosmos3_bench_out/cosmos3_perf.csv
bench-cosmos3:
	bash bench_cosmos3.sh

# Remove benchmark outputs (CSV, per-run JSON, server logs)
bench-cosmos3-clean:
	rm -rf cosmos3_bench_out

# Serve a single checkpoint (foreground; Ctrl-C to stop). vLLM auto-detects the
# quantization format (modelopt) from the checkpoint. For online FP8 from a BF16
# checkpoint, pass EXTRA='--quantization fp8'.
serve-cosmos3:
	$(VLLM_BIN) serve $(CKPT) \
		--port $(SERVE_PORT) \
		--tensor-parallel-size $(TP_SIZE) \
		--gpu-memory-utilization $(GPU_MEM_UTIL) \
		--limit-mm-per-prompt '$(LIMIT_MM)' \
		$(EXTRA)

infer-cosmos3:
	curl -sS http://localhost:8123/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ "model": "/home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/cosmos3-nano/fp8", "messages": [{"role": "user", "content": [ {"type": "text", "text": "What is in this video? Reason about the scene."}, {"type": "video_url", "video_url": {"url": "https://raw.githubusercontent.com/NVIDIA/cosmos/refs/heads/main/cookbooks/cosmos3/reasoner/assets/action_cot_driving_scene.mp4"}} ]}], "max_t okens": 512 }'

build-cuda129:
	DOCKER_BUILDKIT=1 docker build \
		-t $(IMAGE_NAME_CUDA129):$(IMAGE_VERSION) \
		-f docker/Dockerfile.cuda129.local \
		$(PROJECT_ROOT)
	@echo Image $(IMAGE_NAME) built.

upload-cuda129:
	docker tag $(IMAGE_NAME_CUDA129):$(IMAGE_VERSION) nvcr.io/nvidian/vllm-cosmos3:v0.25.0_cu129_git_vllm_$(IMAGE_VERSION) && \
	docker push nvcr.io/nvidian/vllm-cosmos3:v0.25.0_cu129_git_vllm_$(IMAGE_VERSION)

build-cuda129-reasoner:
	DOCKER_BUILDKIT=1 docker build \
		-t $(IMAGE_NAME_CUDA129)-reasoner:$(IMAGE_VERSION) \
		-f docker/Dockerfile.cuda129.local.reasoner \
		$(PROJECT_ROOT)
	@echo Image $(IMAGE_NAME) built.

upload-cuda129-reasoner:
	docker tag $(IMAGE_NAME_CUDA129)-reasoner:$(IMAGE_VERSION) nvcr.io/nvidian/vllm-cosmos3:v0.25.0_reasoner-overlay_cu129_git_vllm_$(IMAGE_VERSION) && \
	docker push nvcr.io/nvidian/vllm-cosmos3:v0.25.0_reasoner-overlay_cu129_git_vllm_$(IMAGE_VERSION)

run-nvcr-cuda129:	
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-p $(HOST_PORT):8000 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		--entrypoint bash \
		nvcr.io/nvidian/vllm-cosmos3:v0.25.0_cu129_git_vllm_$(IMAGE_VERSION)

run-nvcr-cuda129-reasoner:	
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-p $(HOST_PORT):8000 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		--entrypoint bash \
		nvcr.io/nvidian/vllm-cosmos3:v0.25.0_reasoner-overlay_cu129_git_vllm_$(IMAGE_VERSION)
