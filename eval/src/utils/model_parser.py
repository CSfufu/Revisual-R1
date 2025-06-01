import io
import re
import os
import math
import base64
import shutil

from utils.model_wrapper import (
    ModelWrapper
)


def extract_boxed_answer(text):
    """Extract the last boxed answer from generated text, if present."""
    text = text.replace(' \\text{ and } ', ', ') \
               .replace(' \\text{and} ', ', ') \
               .replace(' and ', ', ')

    boxed_matches = re.findall(r'\\boxed{([^}]+)}', text)
    if boxed_matches:
        return boxed_matches[-1].strip(), True  # Return the last match
    return text, False


def get_cache_file_path(cache_dir: str, model_name: str, task) -> str:
    folder = os.path.abspath(os.path.join(cache_dir, f'{model_name}___{task["dataset"]}'))
    return os.path.join(folder, f'index___{task["_index"]:06d}.txt')


def generate_prediction(model: ModelWrapper, task, args) -> str:
    """Generate a prediction for a given task"""
    # print('generate_prediction', task['id'])

    if args.cache_dir is not None:
        cache_file = get_cache_file_path(args.cache_dir, model.model_name, task)
        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    response = f.read()
                    if len(response) > 0:
                        print('generate_prediction', task['_index'], task['id'], 'cache hit')
                        return response
                    else:
                        pass  # need to generate again
            except Exception as e:
                print(f'Error reading cache file {cache_file}: {e}')

    buffer = io.BytesIO()
    image = task['image'][0]
    print(f'<image ({image.width}x{image.height})>')
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

    print('generate_prediction', task['_index'], task['id'])
    response = None
    retries = 0
    max_retries = args.max_retries
    while retries <= max_retries:
        try:
            response = model.generate(messages)
            assert len(response) > 0, 'response is empty'
            break
        except Exception as e:
            if retries < max_retries:
                retries += 1
                print(f'Error generating prediction {task["id"]}, retrying ({retries}/{max_retries}): {e}')
                continue
            else:
                raise RuntimeError(f'Error generating prediction {task["id"]}: {e}')

    if args.cache_dir is not None:
        cache_file = get_cache_file_path(args.cache_dir, model.model_name, task)
        with open(cache_file, 'w+', encoding='utf-8') as f:
            f.write(response)
            f.close()

    return response


def evaluate_prediction(prediction, task, args) -> float:
    prediction_answer, is_boxed = extract_boxed_answer(prediction)
    # assert is_boxed, 'prediction is not a boxed answer'

    answer = task['answer']

    def parse_answer(x): return ''.join(sorted(list(x.strip().lower().replace(',', '').replace(' ', ''))))

    # print('evaluate_prediction', task['id'], f'{prediction_answer}[{parse_answer(prediction_answer)}] | {answer}[{parse_answer(answer)}]')

    if parse_answer(prediction_answer) == parse_answer(answer):
        return 1.0
    else:
        return 0.0
