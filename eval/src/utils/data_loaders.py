import os
import json
from typing import Optional
from datasets import load_dataset


def load_fufu_hard1_dataset(cache_dir: Optional[str] = None):
    ds = load_dataset("csfufu/fufu_hard1", cache_dir=cache_dir)
    result = []
    for i, example in enumerate(ds["train"]):
        example['_index'] = i
        example['dataset'] = 'fufu_hard1'
        result.append(example)
    return result


def load_mathverse_dataset(cache_dir: Optional[str] = None):
    ds = load_dataset("AI4Math/MathVerse", "testmini", cache_dir=cache_dir)
    result = []
    for i, example in enumerate(ds["testmini"]):
        print(i, example)
        example['_index'] = i
        example['id'] = i + 1
        example['dataset'] = 'mathverse'
        example['source'] = 'MathVerse'
        example['image'] = [example['image']]
        example['question'] = example['query_cot']
        del example['query_wo']
        del example['query_cot']
        del example['question_for_eval']
        result.append(example)
    return result


def load_mathvision_dataset(cache_dir: str = None):
    def build_mathvision_prompt(question, choices):
        prompt = 'Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n'
        prompt += 'Question: ' + question + '\n'
        prompt += 'Choices:\n'
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}: {choice}\n"
        return prompt
    
    ds = load_dataset("MathLLMs/MathVision", cache_dir=cache_dir)
    result = []
    for i, example in enumerate(ds["test"]):
        example['_index'] = i
        example['id'] = i + 1
        example['dataset'] = 'mathvision'
        example['source'] = 'MathVision'
        example['image'] = [example['decoded_image']]
        del example['decoded_image']
        example['question'] = build_mathvision_prompt(example['question'], example['options'])
        result.append(example)
    return result


if __name__ == "__main__":
    dirname = os.path.abspath(os.path.dirname(__file__))
    dataset_dir = os.path.join(dirname, "../../dataset")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # ds = load_fufu_hard1_dataset(dataset_dir)
    # ds = load_mathverse_dataset(dataset_dir)
    ds = load_mathvision_dataset(dataset_dir)
