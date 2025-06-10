# groq_llm.py
from llama_index.core.base.llms.base import LLM
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.core.base.llms.types import MessageRole
from typing import List, Optional
from groq import Groq

class GroqLLM(LLM):
    def __init__(self, model: str = "mixtral-8x7b-32768", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self.client = Groq()  # Automatically picks up GROQ_API_KEY from env

    @property
    def metadata(self):
        return {"name": "Groq", "model": self.model}

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return CompletionResponse(text=response.choices[0].message.content)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=True,
        )
        def gen():
            for chunk in stream:
                yield chunk.choices[0].delta.content or ""
        return gen()
