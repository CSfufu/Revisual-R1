from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse

parser = argparse.ArgumentParser(description="Run Revisual-R1 model inference.")
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the model.")
parser.add_argument("--image_path", type=str, default="", help="Path to the input image.")
parser.add_argument("--question", type=str, default="", help="The input question.")
parser.add_argument("--prompt", type=str, default="", help="The input prompt.")
parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens of model generation")
parser.add_argument("--do_sample", type=bool, default=True, help="do_sample of generate")
parser.add_argument("--temperature", type=float, default=0.6, help="Temperature of generate")
parser.add_argument("--top_p", type=float, default=0.95, help="top_p of generate")
parser.add_argument("--top_k", type=int, default=50, help="top_k of generate")
parser.add_argument("--num_return_sequences", type=int, default=1, help="num_return_sequences of generate")
args = parser.parse_args()

file_path = args.model_path

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    file_path, torch_dtype="auto", device_map="auto"
)


min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(file_path, min_pixels=min_pixels, max_pixels=max_pixels)

question = args.question
prompt = args.prompt

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": args.image_path,
            },
            {"type": "text", "text": question + prompt},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)


print("Generating...")
generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, num_return_sequences=args.num_return_sequences)
print("Generation finished.")
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

print("Decoding...")
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

response_token_count = len(generated_ids_trimmed[0])
print(f"Response token count: {response_token_count}")



