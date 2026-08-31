"""
TEI Embeddings — custom LangChain Embeddings wrapper for TEI service.
"""

from typing import List
import requests
from langchain_core.embeddings import Embeddings


class TEIEmbeddings(Embeddings):
    """
    Embeddings class that calls TEI (Text Embeddings Inference) API.
    Compatible with LangChain's Embeddings interface.
    """

    def __init__(self, url: str, api_key: str):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.embed_endpoint = f"{self.url}/embed"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"inputs": texts}

            response = requests.post(
                self.embed_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                raise ValueError(f"TEI embedder error: {response.status_code} - {response.text[:200]}")

            return response.json()

        except Exception as e:
            print(f"ERROR: TEI embeddings failed: {e}", flush=True)
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        result = self.embed_documents([text])
        return result[0] if result else []
