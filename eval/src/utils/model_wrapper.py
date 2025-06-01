import abc
import openai
from typing import List


class ModelWrapper(abc.ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, repetition_penalty: float) -> str:
        pass


class OpenAIWrapper(ModelWrapper):
    def __init__(self,
                 model_name: str,
                 api_key: str,
                 base_url: str,
                 max_tokens: int,
                 temperature: float,
                 top_p: float,
                 ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, messages: List[dict], stream: bool = True) -> str:
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

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
                    # print(chunk.choices[0].delta.content, end='')
                    content += chunk.choices[0].delta.content
                else:
                    # print('None content:', chunk)
                    pass
            return content
        else:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return response.choices[0].message.content
