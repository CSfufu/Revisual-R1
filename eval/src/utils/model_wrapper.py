import os
import abc
import json
import openai
import hashlib
from typing import List, Union, Optional


def hash_json_obj(obj: Union[dict, list]) -> str:
    try:
        json_str = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        json_bytes = json_str.encode('utf-8')
        return hashlib.sha256(json_bytes).hexdigest()
    except TypeError as e:
        print(f'Error: Data structure is not JSON serializable: {e}')
        raise ValueError(f'Data structure is not JSON serializable: {e}')


class ModelWrapper(abc.ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, repetition_penalty: float) -> str:
        pass


class OpenAIWrapper(ModelWrapper):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
    ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.settings = {
            "api_key": api_key,
            "base_url": base_url,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        self.max_retries = max_retries

        self.model_hash = model_name + '_' + hash_json_obj(self.settings)[:6]
        self.cache_dir = os.path.join(cache_dir, 'predictions', self.model_hash)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def generate(self, messages: List[dict], stream: bool = True) -> str:
        if self.cache_dir is not None:
            # check if cached response exists
            message_hash = hash_json_obj(messages)
            cache_file = os.path.join(self.cache_dir, f'{message_hash}.txt')
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf8') as f:
                    prediction = f.read()
                    assert len(prediction) > 0, 'response is empty'
                    return prediction

        print(f'generate stream={stream}')

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        retries = 0
        while retries <= self.max_retries:
            try:
                if stream:
                    content = ""
                    for chunk in client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stream=True
                    ):
                        if chunk.choices[0].delta.content is not None:
                            print(chunk.choices[0].delta.content, end='')
                            content += chunk.choices[0].delta.content
                        else:
                            # print('None content:', chunk)
                            pass
                    prediction = content
                else:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    prediction = response.choices[0].message.content
                    assert len(prediction) > 0, 'response is empty'
                break
            except Exception as e:
                if retries < self.max_retries:
                    retries += 1
                    print(f'Error generating prediction, retrying ({retries}/{self.max_retries}): {e}')
                    continue
                else:
                    raise RuntimeError(f'Error generating prediction: {e}')

        if self.cache_dir is not None:
            message_hash = hash_json_obj(messages)
            cache_file = os.path.join(self.cache_dir, f'{message_hash}.txt')
            with open(cache_file, 'w', encoding='utf8') as f:
                f.write(prediction)

        return prediction
