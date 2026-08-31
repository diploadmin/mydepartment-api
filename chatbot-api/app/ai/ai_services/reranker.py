"""
TEI Reranker — cross-encoder reranking via TEI API.
"""

from typing import List
import requests
from langchain_core.documents import Document


class TEIReranker:
    """
    Reranker that calls TEI (Text Embeddings Inference) reranker API.
    Uses cross-encoder models for more accurate relevance scoring.
    """

    def __init__(self, url: str, api_key: str, top_k: int = 8):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.top_k = top_k
        self.rerank_endpoint = f"{self.url}/"

    def rerank_sentences(self, query: str, sentences: List[str]) -> List[dict]:
        """
        Score individual sentences against a query using TEI cross-encoder.

        Returns:
            List of {"index": int, "score": float} sorted by score descending.
            On failure returns empty list.
        """
        if not sentences:
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"query": query, "texts": sentences}

            response = requests.post(
                self.rerank_endpoint, headers=headers,
                json=payload, timeout=5
            )

            if response.status_code != 200:
                print(f"WARNING: Sentence reranker returned status {response.status_code}", flush=True)
                return []

            results = response.json()
            if isinstance(results, list):
                sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            else:
                sorted_results = sorted(results.get('results', results),
                                        key=lambda x: x.get('score', 0), reverse=True)

            return [{"index": int(r.get("index", 0)), "score": r.get("score", 0.0)}
                    for r in sorted_results]
        except Exception as e:
            print(f"WARNING: Sentence reranker error: {e}", flush=True)
            return []

    def rerank(self, query: str, documents: List[Document], top_k: int = 0) -> List[Document]:
        """
        Rerank documents using TEI reranker API.

        Args:
            query: The search query
            documents: List of Document objects to rerank
            top_k: Override instance top_k (0 = use self.top_k)

        Returns:
            Reranked list of documents (top_k)
        """
        effective_top_k = top_k if top_k > 0 else self.top_k
        if not documents:
            return documents

        MAX_TEXT_LEN = 4000

        texts = []
        structured_count = 0
        for doc in documents:
            best_sentence = doc.metadata.get('_best_sentence', '')
            title = doc.metadata.get('title', '') or doc.metadata.get('h1', '')
            section_title = doc.metadata.get('_section_title', '')
            label = doc.metadata.get('label', '')

            display_title = title
            if section_title and section_title.lower() != (title or "").lower():
                display_title = f"{title} > {section_title}" if title else section_title

            title_prefix = ""
            if display_title:
                title_prefix = f"[{label}] {display_title}\n\n" if label else f"{display_title}\n\n"

            if best_sentence and best_sentence != doc.page_content:
                context = doc.page_content
                if len(context) > MAX_TEXT_LEN - 500:
                    context = context[:MAX_TEXT_LEN - 500] + "..."
                structured_text = f"{title_prefix}[BEST MATCH]: {best_sentence}\n\n[FULL CONTEXT]: {context}"
                texts.append(structured_text)
                structured_count += 1
            else:
                text = f"{title_prefix}{doc.page_content}"
                if len(text) > MAX_TEXT_LEN:
                    text = text[:MAX_TEXT_LEN] + "..."
                texts.append(text)

        print(f"DEBUG: Reranker using structured format for {structured_count}/{len(documents)} documents", flush=True)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"query": query, "texts": texts}

            response = requests.post(
                self.rerank_endpoint,
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code != 200:
                print(f"WARNING: Reranker returned status {response.status_code}: {response.text[:200]}", flush=True)
                return documents[:effective_top_k]

            results = response.json()

            if isinstance(results, list):
                sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            else:
                sorted_results = sorted(results.get('results', results), key=lambda x: x.get('score', 0), reverse=True)

            reranked_docs = []
            for item in sorted_results[:effective_top_k]:
                idx = int(item.get('index', 0))
                if idx < len(documents):
                    doc = documents[idx]
                    doc.metadata['_reranker_score'] = item.get('score', 0)
                    reranked_docs.append(doc)

            print(f"DEBUG: Reranker returned {len(reranked_docs)} documents (from {len(documents)} candidates, top_k={effective_top_k})", flush=True)
            return reranked_docs

        except requests.exceptions.Timeout:
            print("WARNING: Reranker request timed out, returning original documents", flush=True)
            return documents[:effective_top_k]
        except Exception as e:
            print(f"WARNING: Reranker error: {e}, returning original documents", flush=True)
            return documents[:effective_top_k]
