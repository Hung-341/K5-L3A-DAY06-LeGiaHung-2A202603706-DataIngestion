"""Create real embeddings using an OpenAI-compatible API."""
import json
import os
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class MyEmbedder:
    """OpenAI-compatible embedding adapter."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name.strip()
        self.base_url = os.getenv(
            "EMBEDDING_BASE_URL",
            "https://api.openai.com/v1",
        ).rstrip("/")
        self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "0"))

        self.api_key = (
            os.getenv("EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        )

        self.total_tokens = 0  # usage.total_tokens reported by the provider

        if not self.model_name:
            raise ValueError("Thiếu EMBEDDING_MODEL trong .env.")
        if not self.api_key:
            raise ValueError(
                "Thiếu EMBEDDING_API_KEY, OPENAI_API_KEY hoặc OPENROUTER_API_KEY."
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = json.dumps({
            "model": self.model_name,
            "input": texts,
        }).encode("utf-8")

        request = Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        for attempt in range(4):
            try:
                with urlopen(request, timeout=90) as response:
                    body = json.loads(response.read().decode("utf-8"))

                rows = sorted(body["data"], key=lambda row: row["index"])
                vectors = [
                    [float(value) for value in row["embedding"]]
                    for row in rows
                ]

                if len(vectors) != len(texts):
                    raise ValueError(
                        "Provider trả số vector không khớp số văn bản."
                    )
                if not vectors or any(not vector for vector in vectors):
                    raise ValueError("Provider trả vector rỗng.")

                observed_dimensions = len(vectors[0])
                if any(len(vector) != observed_dimensions for vector in vectors):
                    raise ValueError("Kích thước vector trong response không đồng nhất.")

                if self.dimensions and observed_dimensions != self.dimensions:
                    raise ValueError(
                        f"Đã cấu hình {self.dimensions} chiều nhưng API trả "
                        f"{observed_dimensions} chiều."
                    )

                self.dimensions = observed_dimensions
                self.total_tokens += int(
                    (body.get("usage") or {}).get("total_tokens", 0)
                )
                return vectors

            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                if exc.code != 429 and exc.code < 500:
                    raise RuntimeError(
                        f"Embedding API trả HTTP {exc.code}: {detail}"
                    ) from exc
                if attempt == 3:
                    raise RuntimeError(
                        f"Embedding API vẫn lỗi sau nhiều lần thử: {detail}"
                    ) from exc

            except URLError as exc:
                if attempt == 3:
                    raise RuntimeError(f"Không kết nối được embedding API: {exc}") from exc

            time.sleep(2**attempt)

        raise RuntimeError("Không lấy được embedding.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in batches, preserving input order."""
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
        if batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE phải lớn hơn 0.")

        vectors = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._embed(texts[start : start + batch_size]))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        """Embed one query using the same model as the documents."""
        vectors = self._embed([question])
        return vectors[0]