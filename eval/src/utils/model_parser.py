import io
import re
import os
import math
import base64
from typing import Optional

from utils.model_wrapper import (
    ModelWrapper
)




def build_extract_prompt(prediction, question):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt


def extract_boxed_answer(text):
    """Extract the last boxed answer from generated text, if present."""
    text = text.replace(' \\text{ and } ', ', ') \
               .replace(' \\text{and} ', ', ') \
               .replace(' and ', ', ')

    boxed_matches = re.findall(r'\\boxed{([^}]+)}', text)
    if boxed_matches:
        return boxed_matches[-1].strip(), True  # Return the last match
    return text, False


def generate_prediction(model: ModelWrapper, task, args) -> str:
    """Generate a prediction for a given task"""
    # print('generate_prediction', task['id'])

    buffer = io.BytesIO()
    image = task['image'][0]
    # print(f'<image ({image.width}x{image.height})>')
    if (image.width * image.height) > args.max_pixels:
        resize_factor = math.sqrt(args.max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
    if (image.width * image.height) < args.min_pixels:
        resize_factor = math.sqrt(args.min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    messages = [
        {
            "role": "system",
            "content": args.system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": task['question']
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ],
        }
    ]

    return model.generate(messages)


def evaluate_prediction(prediction, task, args, model: Optional[ModelWrapper] = None) -> float:
    def parse_options(x): return ''.join(sorted(list(x.strip().lower().replace(',', '').replace(' ', ''))))
    def parse_alphas(x): return ''.join(filter(lambda c: c.isalpha(), list(x)))

    def parse_answer(x):
        extracted, is_boxed = extract_boxed_answer(x)
        # print('extracted', extracted, is_boxed, '=>', parse_alphas(extracted))
        return parse_alphas(extracted)

    prediction_answer, is_boxed = extract_boxed_answer(prediction)
    if is_boxed:
        # print('evaluate_prediction', task['id'], f'{prediction_answer}[{parse_answer(prediction_answer)}] | {task['answer']}[{parse_answer(task['answer'])}]')
        if parse_options(prediction_answer) == parse_options(task['answer']):
            return 1.0

    # use llm to judge
    if model is None:
        return 0.0

    prompt = build_extract_prompt(prediction, task['question'])
    messages = [
        {
            "role": "system",
            "content": args.system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    extracted_answer = model.generate(messages)
    extracted_answer = re.sub(r'<think>.*?</think>', '', extracted_answer, flags=re.DOTALL)

    # print('llm extract:', parse_answer(extracted_answer), parse_answer(task['answer']))
    if parse_answer(extracted_answer) == parse_answer(task['answer']):
        return 1.0

    return 0.0
