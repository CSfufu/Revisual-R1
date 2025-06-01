import os
import json
from datasets import load_dataset


def load_fufu_hard1_dataset(cache_dir: str = None):
    if cache_dir is None:
        ds = load_dataset("csfufu/fufu_hard1")
    else:
        ds = load_dataset("csfufu/fufu_hard1", cache_dir=cache_dir)
    result = []
    for i, example in enumerate(ds["train"]):
        example['_index'] = i
        example['dataset'] = 'fufu_hard1'
        result.append(example)
    return result


if __name__ == "__main__":
    dirname = os.path.abspath(os.path.dirname(__file__))
    dataset_dir = os.path.join(dirname, "../../dataset")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    ds = load_fufu_hard1_dataset(dataset_dir)

    max_examples = 5
    for i, example in enumerate(ds["train"]):
        if i < max_examples:
            print(f"--- Example #{i} ---")
            for key, value in example.items():
                if key == "image":
                    print(f"{key}: {value[0]}")
                else:
                    print(f"{key}: {value}")
            print("\n")
        else:
            break
