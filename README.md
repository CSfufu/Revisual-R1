# Revisual-R1

## Abstract

Recently, Deepseek-R1 has demonstrated remarkable
reasoning capability in complex textual tasks with reinforcement learning (RL). 
To incentivize similar capability in Multimodal Large Language Models (MLLMs), most existing methods directly apply RL to them, however, struggle to activate complex reasoning capabilities. 
In this paper, instead of isolately studying multimodal RL, we dig into current training pipelines and identify three commonly overlooked phenomenons: 
1) Sufficient cold start initialization can effectively improve the reasoning ability of MLLMs. Intriguingly, with carefully selected text data as a cold start, our model surpasses most recent multimodal reasoning models. 
2) Standard GRPO for multimodal RL suffers from gradient stagnation, therefore degrading the training stability and performance. 
3) After multimodal RL, post-text RL training can further improve the multimodal reasoning ability. This iterative multimodal and text RL training strategy effectively balances the reasoning ability for perception and cognition. 
By incorporating the above observations and addressing the issues in multimodal RL, we introduce ReVisual‑R1,  
which sets a new state‑of‑the‑art among open‑source 7B MLLMs on challenging benchmarks, including MathVerse, MathVision, WeMath, LogicVista, and DynaMath. 

## 🛠️ Usage
### (Step1) Install
```bash
conda create -n revisual python=3.11 -y && conda activate revisual

cd Revisual-R1
pip3 install -e .
```

### (Step 2) Training
```bash
bash ./examples/format_reward.sh
```
If you encounter issues with connecting to Hugging Face, consider using export HF_ENDPOINT=https://hf-mirror.com.


### (Step 3) Merge Checkpoint in Hugging Face Format
```bash
python3 scripts/model_merger.py --local_dir checkpoints/${ProjectName}$/exp_name/global_step_1/actor
```

### (Step 4) Evaluation

### Usage

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

### Examples

**(1)** Evaluate a model directly via OpenAI API

```shell
python ./src/main.py --model-name="gpt-4.1" \
	--openai-api-key="YOUR_API_KEY" \
	--cache-dir="./cache"
```

**(2)** Deploy and evaluate a local model using [lmdeploy](https://github.com/InternLM/lmdeploy)

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
