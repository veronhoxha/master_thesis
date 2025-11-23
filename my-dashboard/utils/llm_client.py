import os
from typing import Optional


class LLMClient:
    def __init__(self, model: Optional[str] = None):
        # Lazily set up a client only if API key is present
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    def generate_csv(self, prompt: str) -> str:
        """Return pure CSV text. Fallback to echo prompt if client not available."""
        if not self._client:
            # Fallback: return an empty CSV when no client is configured
            return ""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data generator that outputs CSV only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            content = resp.choices[0].message.content or ""
            # Strip code fences if present
            content = content.strip()
            if content.startswith("```") and content.endswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content.rsplit("\n", 1)[0]
            return content
        except Exception as e:
            return ""


