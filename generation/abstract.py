from abc import ABC, abstractmethod
import re


class GenerativeModel(ABC):
    def __init__(self, model_name,
                 prompt_model=None):
        self.model_name = model_name
        self.prompt_model = prompt_model

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def extract_code(self, result):
        pattern = r"```(.*?)```"
        match = re.search(pattern, result, re.DOTALL)
        if match:
            result = match.group(1)
        return result

