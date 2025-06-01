set -x

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="xxx"

MODEL_PATH=xxx  # replace it with your local file path
EXPERIMENT_NAME=xxx
PROJECT_NAME=xxx
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"



python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="train_dataset"@train \
    data.val_files="value_dataset"@test \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    algorrithm.use_pad=true \
    algorithm.pad_alpha=1.5 \
    algorithm.pad_advantage_threshold_low=0.03 \
    algorithm.pad_advantage_threshold_high=0.97 \
    algorithm.pad_sample_size_ratio=1.5 \
    algorithm.pad_with_replacement=true \
    algorithm.pad_normalize_by_length=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.entropy_coef_init=0.02 \
    worker.actor.entropy_coef_min=0.0 \
    worker.actor.entropy_decay=0.985 \
    worker.actor.entropy_schedule="exp" \
    worker.actor.total_updates=200000 \
    worker.actor.entropy_warmup_steps=20 \
    worker.optim.lr=1e-6 \
    worker.rollout.n=8 \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.val_override_config.temperature=0.3 \
    worker.reward.use_efficient_reward=false \
    worker.reward.target_max_length=8192 \
    trainer.max_steps=200 \
    trainer.save_limit=5 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.n_gpus_per_node=8
