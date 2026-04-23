from prompt.prompt_model import PromptModel
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation.abstract import GenerativeModel
from utils.logging_config import log_llm_prompt, log_llm_response


class CausalLM(GenerativeModel):
    def __init__(self, model_name, huggingface_api_key=None, prompt_model=None):
        super().__init__(model_name, prompt_model)
        if prompt_model is None:
            self.prompt_model = PromptModel()

        self.model = AutoModelForCausalLM.from_pretrained(
            f'{model_name}',
            torch_dtype="auto",
            device_map="auto",
            token=huggingface_api_key
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            f'{model_name}',
            token=huggingface_api_key
        )

    def generate(self,
                 model_prompt_dir: str,
                 prompt_name: str,
                 prefix: Optional[str] = None,
                 test: Optional[bool] = False,
                 no_code_extract: bool = False,
                 **replacements) -> str:

        system_prompt, user_prompt = self.prompt_model.process_prompt(
            model_name=model_prompt_dir,
            prompt_name=prompt_name,
            **replacements
        )

        log_llm_prompt(prompt_name, system_prompt, user_prompt)

        result = None
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=4096,
                temperature=0.01,
                do_sample=True
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            log_llm_response(prompt_name, result)

            if '</think>' in result:
                result = result.split('</think>')[-1]

            if test:
                print(result)

            if not no_code_extract and "```" in result:
                result = self.extract_code(result)

            if prefix and prefix in result:
                parts = result.split(prefix, 1)
                if len(parts) > 1:
                    result = parts[1].strip()
            return result

        except Exception as e:
            print(e)
            return None
