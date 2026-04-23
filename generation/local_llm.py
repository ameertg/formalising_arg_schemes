"""
Local LLM support using HuggingFace Transformers.

This module provides a wrapper for running local language models via HuggingFace,
optimized for macOS including Apple Silicon.

Installation:
    pip install transformers torch accelerate

Recommended models for MacBook Pro:
    - Qwen/Qwen2.5-3B-Instruct (3B parameters, good quality)
    - Qwen/Qwen2.5-1.5B-Instruct (1.5B parameters, very fast)
    - microsoft/Phi-3-mini-4k-instruct (3.8B parameters, good reasoning)
    - google/gemma-2-2b-it (2B parameters, efficient)
"""

from prompt.prompt_model import PromptModel
from typing import Optional
from generation.abstract import GenerativeModel
from utils.logging_config import log_llm_prompt, log_llm_response


class LocalLLM(GenerativeModel):
    """Local LLM using HuggingFace Transformers."""

    def __init__(self, model_name: str, device: str = "mps", prompt_model=None,
                 load_in_8bit: bool = False):
        """
        Initialize local LLM with HuggingFace.

        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B-Instruct")
            device: Device to run on ("mps" for Apple Silicon, "cpu" for Intel, "cuda" for Nvidia)
            prompt_model: Optional prompt model for prompt processing
            load_in_8bit: If True, load model in 8-bit quantization (requires bitsandbytes, CUDA only)
        """
        super().__init__(model_name, prompt_model)
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.pipeline = None
        self.tokenizer = None
        self.processor = None  # used for Gemma models
        self.model = None
        self._is_gemma = 'gemma' in model_name.lower()

        if prompt_model is None:
            self.prompt_model = PromptModel()

        # Lazy loading - only load when first used
        self._initialized = False

    def _initialize_model(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        except ImportError as e:
            raise RuntimeError(
                f"PyTorch import error: {e}\n"
                "Install with: pip install torch"
            )

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, pipeline
            import transformers
            print(f"Transformers version: {transformers.__version__}")
        except ImportError as e:
            raise RuntimeError(
                f"Transformers import error: {e}\n"
                "Install with: pip install transformers accelerate"
            )

        print(f"Loading model {self.model_name}...")
        print(f"This may take a few minutes on first run...")

        # Determine device
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            self.device = "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"

        print(f"Using device: {self.device}")

        # Load tokenizer/processor and model
        try:
            if self._is_gemma:
                print("Detected Gemma model — using AutoProcessor")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype="auto",
                    device_map="auto"
                )
                self._initialized = True
                print(f"✓ Gemma model loaded successfully")
                return

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Load model with appropriate settings for device
            if self.device == "mps":
                # Apple Silicon
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="mps",
                    trust_remote_code=True
                )
            elif self.device == "cpu":
                # CPU fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model = self.model.to("cpu")
            else:
                # CUDA or other
                load_kwargs = {
                    "device_map": "auto",
                    "trust_remote_code": True,
                }
                if self.load_in_8bit:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    print("Loading model in 8-bit quantization...")
                else:
                    load_kwargs["torch_dtype"] = torch.float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_kwargs
                )

            # Create pipeline with proper defaults
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device in ["cpu", "mps"] else None,
                max_length=None,  # Disable default max_length
                max_new_tokens=4096  # Set our default
            )

            print(f"✓ Model loaded successfully on {self.device}")
            self._initialized = True

        except Exception as e:
            raise RuntimeError(
                f"Error loading model {self.model_name}: {e}\n"
                f"Try a smaller model like 'Qwen/Qwen2.5-1.5B-Instruct'"
            )

    def completion(self, messages: list, temperature: float = 0.0,
                   max_tokens: int = 2048) -> str:
        """
        Generate completion using HuggingFace model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        self._initialize_model()

        if self._is_gemma:
            try:
                import torch
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                    )
                result = self.processor.decode(
                    outputs[0][input_len:], skip_special_tokens=True)
                return result.strip()
            except Exception as e:
                raise RuntimeError(f"Error during Gemma generation: {e}")

        # Format messages using chat template if available
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback: manual formatting
                prompt = ""
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    if role == "system":
                        prompt += f"System: {content}\n\n"
                    elif role == "user":
                        prompt += f"User: {content}\n\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n\n"
                prompt += "Assistant: "

        except Exception as e:
            print(f"Warning: Error formatting prompt: {e}")
            # Simple fallback
            prompt = messages[-1]['content']

        # Generate
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,  # Avoid exact 0
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )

            result = outputs[0]['generated_text']
            return result.strip()

        except Exception as e:
            raise RuntimeError(f"Error during generation: {e}")

    def generate(self,
                 model_prompt_dir: str,
                 prompt_name: str,
                 prefix: Optional[str] = None,
                 test: Optional[bool] = False,
                 no_code_extract: bool = False,
                 **replacements) -> str:
        """
        Generate a response from the local LLM.

        Parameters:
        - model_prompt_dir (str): The directory of the model prompt.
        - prompt_name (str): The name of the prompt.
        - prefix (str, optional): A prefix to extract content after.
        - test (bool, optional): If True, print the raw response.
        - **replacements: Keyword arguments for prompt template replacement.

        Returns:
        - str: Generated and post-processed response.
        """
        # Process prompt templates
        system_prompt, user_prompt = self.prompt_model.process_prompt(
            model_name=model_prompt_dir,
            prompt_name=prompt_name,
            **replacements
        )

        log_llm_prompt(prompt_name, system_prompt, user_prompt)

        # Generate
        try:
            result = self.completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=4096
            )
        except Exception as e:
            print(f'Error generating with local LLM: {e}')
            raise

        log_llm_response(prompt_name, result)

        if test:
            print(result)

        # Post-processing
        if not no_code_extract and "```" in result:
            result = self.extract_code(result)

        if prefix and prefix in result:
            parts = result.split(prefix, 1)
            if len(parts) > 1:
                result = parts[1].strip()

        return result


if __name__ == "__main__":
    # Test script
    print("Testing LocalLLM with HuggingFace...")

    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Determine best device
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"Using Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"Using NVIDIA GPU (CUDA)")
        else:
            device = "cpu"
            print(f"Using CPU")

        # Test with a small model
        print(f"\nTesting with Qwen/Qwen2.5-1.5B-Instruct...")
        llm = LocalLLM("Qwen/Qwen2.5-1.5B-Instruct", device=device)

        response = llm.completion([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' and explain what this phrase means in one sentence."}
        ])
        print(f"\nResponse: {response}")
        print("\n✓ LocalLLM is working correctly!")

    except ImportError:
        print("\nRequired packages not installed.")
        print("Install with: pip install transformers torch accelerate")
    except Exception as e:
        print(f"\n✗ Error: {e}")
