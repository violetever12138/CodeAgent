"""Optional, local-only Qwen encoder with distinct query/document contracts."""

from __future__ import annotations

import json
from threading import RLock

QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_INSTRUCTION = "Given a code search query, retrieve relevant code snippets from the repository."


class QwenEmbeddingProvider:
    model_id = QWEN_MODEL
    dimension = 1024

    def __init__(self, *, revision=QWEN_REVISION, device="cpu", max_tokens=8192, batch_size=32):
        if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Qwen revision must be a fixed commit")
        if not 0 < max_tokens <= 32768 or batch_size <= 0:
            raise ValueError("invalid Qwen token or batch limit")
        self.revision, self.device = revision, device
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self._model = self._tokenizer = None
        self._lock = RLock()

    def cache_identity(self, role):
        return json.dumps({"schema": "qwen-code-v1", "model": self.model_id,
                           "revision": self.revision, "dimension": self.dimension,
                           "role": role, "instruction": QUERY_INSTRUCTION if role == "query" else None,
                           "max_tokens": self.max_tokens, "pooling": "last-token", "normalize": "l2",
                           "dtype": "bf16" if self.device.startswith("cuda") else "fp32"}, sort_keys=True)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, padding_side="left", local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=True,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        ).to(self.device).eval()

    @staticmethod
    def last_token_pool(hidden, mask):
        import torch
        if bool((mask[:, -1].sum() == mask.shape[0]).item()):
            return hidden[:, -1]
        lengths = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]

    def _encode(self, texts, role):
        if not texts:
            return []
        import torch
        import torch.nn.functional as functional
        with self._lock, torch.inference_mode():
            self._load()
            inputs = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}" if role == "query" else text
                      for text in texts]
            vectors = []
            for offset in range(0, len(inputs), self.batch_size):
                batch = self._tokenizer(inputs[offset:offset + self.batch_size], padding=True,
                                        truncation=True, max_length=self.max_tokens, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                hidden = self._model(**batch).last_hidden_state
                pooled = self.last_token_pool(hidden, batch["attention_mask"])
                if pooled.shape[-1] != self.dimension:
                    raise ValueError("Qwen embedding dimension is not 1024")
                vectors.extend(functional.normalize(pooled.float(), p=2, dim=1).cpu().tolist())
            return vectors

    def embed_queries(self, texts):
        return self._encode(texts, "query")

    def embed_documents(self, texts):
        return self._encode(texts, "document")
