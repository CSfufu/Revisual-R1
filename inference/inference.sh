# This script is used to run inference using a pre-trained model specified by MODEL_PATH.
# It sets various parameters for the inference process, such as the maximum number of tokens,
# sampling options, and the prompt format. The script then calls a Python script (infer.py)
# with these parameters to generate predictions based on the input question and image.

MODEL_PATH="Reviusal-R1"
MAX_TOKENS=16384
DO_SAMPLE=True
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=50
NUM_RETURN_SEQUENCES=1


prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
question="Which of the boxes comes next in the sequence? Select answers from A-E"


python infer.py \
 --model_path ${MODEL_PATH} \
 --image_path ${IMAGE_PATH} \
 --question ${question} \
 --prompt ${prompt} \
 --max_tokens ${MAX_TOKENS} \
 --do_sample ${DO_SAMPLE} \
 --temperature ${TEMPERATURE} \
 --top_p ${TOP_P} \
 --top_k ${TOP_K} \
 --num_return_sequences ${NUM_RETURN_SEQUENCES} 