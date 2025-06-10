# groq_llm.py
from typing import Optional, List, Generator
from llama_index.llms.base import LLM
from llama_index.llms.base import CompletionResponse, CompletionResponseGen
from groq import Groq


class GroqLLM(LLM):
    def __init__(self, model: str = "mixtral-8x7b-32768", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self.client = Groq()

    @property
    def metadata(self):
        return {
            "context_window": 8192,
            "num_output": 512,
            "is_chat_model": True,
            "model_name": f"groq/{self.model}",
        }

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return CompletionResponse(text=res.choices[0].message.content)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=True,
        )

        def gen() -> Generator[str, None, None]:
            for chunk in stream:
                yield chunk.choices[0].delta.content or ""

        return gen()
