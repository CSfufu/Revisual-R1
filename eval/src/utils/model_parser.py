import io
import re
import os
import math
import base64
from typing import Optional

from utils.model_wrapper import (
    ModelWrapper
)


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]



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
