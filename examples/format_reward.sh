set -x

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="162896d930225822f8a4d78fbe4bfd4f05bff14c"

MODEL_PATH=/map-vepfs/tomiyasu/Datasets/Qwen2.5-VL-7B-Instruct-Coldstart  # replace it with your local file path
EXPERIMENT_NAME=format_reward
PROJECT_NAME=FormalMMRL
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/map-vepfs/tomiyasu/Reasoning/EasyR1/datasets/geometry3k@train \
    data.val_files=/map-vepfs/tomiyasu/Reasoning/EasyR1/datasets/geometry3k@test \
    data.max_response_length=4096 \
    data.max_prompt_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.n_gpus_per_node=4
