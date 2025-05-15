## eval

> [English](./README.md) | [简体中文](./README.zh_CN.md)

### 用法

```plain
usage: main.py [-h] --model-name MODEL_NAME --openai-api-key OPENAI_API_KEY [--openai-base-url OPENAI_BASE_URL] [--cache-dir CACHE_DIR] [--output-dir OUTPUT_DIR] [--max-tokens MAX_TOKENS] [--min-pixels MIN_PIXELS]
               [--max-pixels MAX_PIXELS] [--temperature TEMPERATURE] [--top-p TOP_P] [--system-prompt SYSTEM_PROMPT] [--datasets DATASETS] [--dataset-dir DATASET_DIR] [--eval-threads EVAL_THREADS] [--max-retries MAX_RETRIES]

Unified evaluation for multimodal math datasets

options:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        The name of the model to use
  --openai-api-key OPENAI_API_KEY
                        The API key for the OpenAI API
  --openai-base-url OPENAI_BASE_URL
                        The base URL for the OpenAI API
  --cache-dir CACHE_DIR
                        Directory to cache predictions
  --output-dir OUTPUT_DIR
                        Directory to save results
  --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate
  --min-pixels MIN_PIXELS
  --max-pixels MAX_PIXELS
  --temperature TEMPERATURE
                        Sampling temperature
  --top-p TOP_P         Top-p sampling
  --system-prompt SYSTEM_PROMPT
                        System prompt for the model
  --datasets DATASETS   Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision or 'all'
  --dataset-dir DATASET_DIR
  --eval-threads EVAL_THREADS
                        Number of threads for evaluation
  --max-retries MAX_RETRIES
                        Maximum number of retries for evaluation
```

### 例子

**(1)** 直接通过 OpenAI API 评测模型

```shell
python ./src/main.py --model-name="gpt-4.1" \
	--openai-api-key="YOUR_API_KEY" \
	--cache-dir="./cache"
```

**(2)** 通过 [lmdeploy](https://github.com/InternLM/lmdeploy) 部署本地模型并评测

```shell
lmdeploy serve api_server \
	/path/to/local/lmm \
	--model-name lmm_name \
	--server-port 23333 \
	--chat-template qwen2d5-vl

python ./src/main.py --model-name="lmm_name" \
	--openai-base-url="http://localhost:23333/v1" \
	--openai-api-key="YOUR_API_KEY" \
	--cache-dir="./cache"
```
