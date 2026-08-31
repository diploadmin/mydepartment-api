from datetime import datetime
from app.core.logging import logger
from app.core.singleton import Singleton
# config that comunicates with .env file
from app.core.config import USE_SOURCE_FILTER, CITE_SOURCES, DEEP_LINK_HIGHLIGHT_MODE, SENTENCE_HIGHLIGHT_MODE, USE_SHORT_DEEP_LINKS, DEEP_LINK_API_URL, USE_CUSTOM_SYSTEM_PROMPT, PDF_PROXY_BASE_URL, DOC_VIEWER_BASE_URL, HIGHLIGHT_SCRIPT_HOSTS
from app.core.retrieval_context import set_retrieval_overrides, clear_retrieval_overrides, get_param
from app.services.deep_link_service import DeepLinkService
from app.models.chat_domain import *
from app.schemas.chat_schema import *
from app.repositories.chat_repository import ChatRepository
from app.repositories.response_repository import ResponseRepository
from fastapi import HTTPException, WebSocket
import json
import ast
import re
import requests as http_requests
from urllib.parse import quote, quote_plus, urlparse
from langchain_core.messages import SystemMessage

from app.services.chatService import *
from app.services.user_type_service import get_system_prompt_by_user_type
# langfuse tracing
from app.core.langfuse_tracing import (
    build_langfuse_run_config,
    build_thread_config,
    get_langfuse_callback_handler,
    langfuse_invoke_config,
    patch_langfuse_trace_metadata,
)
from app.ai.ai_services.retrievers.utils import get_content_filter_debug_info

# this controls layer between the frontend and the backend

# description: _filter_langfuse_metadata - this is a helper that overrides metadata of langfuse with customized WP admin settings
# parameters:
# - overrides: dict - custom WP admin settings with changed metadata for langfuse
# returns:
# - dict - the filtered langfuse metadata
def _filter_langfuse_metadata(overrides: dict) -> dict:
    """Same filter snapshot as debug logs, for Langfuse trace metadata."""
    filters_debug = get_content_filter_debug_info()
    filters_active = bool(filters_debug) or any(
        overrides.get(k)
        for k in (
            "parent_filter_name",
            "site_filter_name",
            "post_types",
            "custom_filters",
            "person_filter_name",
            "person_filter_names",
            "city_filter_name",
            "city_filter_names",
            "country_filter_name",
            "country_filter_names",
            "organisation_filter_name",
            "organisation_filter_names",
            "access_group_name",
            "access_group_names",
        )
    )
    return {
        "filters_active": filters_active,
        "filters_debug": filters_debug,
        "retrieval_overrides": overrides,
    }

# description: either call the custom system prompt or the default system prompt
def _resolve_system_prompt(data) -> str:
    """Return the custom prompt from the request when USE_CUSTOM_SYSTEM_PROMPT is
    enabled and the frontend actually sent one; otherwise fall back to the
    hardcoded role-based prompt."""
    if USE_CUSTOM_SYSTEM_PROMPT and getattr(data, 'system_prompt', None):
        return data.system_prompt
    return get_system_prompt_by_user_type(data.user_type)


# =============================================================================
# URL REDIRECT RESOLVER
# Resolves 301/302 redirects so deep link query params aren't lost.
# Results are cached in-memory for the lifetime of the process.
# =============================================================================
# cache for the url redirects
_url_redirect_cache: dict[str, str] = {}

# description: resolve the url redirects
def resolve_url_redirects(url: str) -> str:
    """
    Follow HTTP redirects and return the final URL.
    Caches results to avoid repeated network requests.
    Preserves any fragment (#anchor) from the redirect target.
    """
    # 
    if url in _url_redirect_cache:
        return _url_redirect_cache[url]
    try:
        resp = http_requests.head(url, allow_redirects=True, timeout=5)
        final_url = resp.url
        # Strip trailing query params from the resolved URL (we'll add our own)
        if '?' in final_url:
            final_url = final_url.split('?')[0]
        _url_redirect_cache[url] = final_url
        logger.info(f"URL redirect resolved: {url} -> {final_url}")
        return final_url
    except Exception as e:
        logger.warning(f"URL redirect resolution failed for {url}: {e}")
        _url_redirect_cache[url] = url
        return url

# description: normalize the url with the redirect
def _normalize_url_with_redirect(url: str) -> tuple[str, str]:
    """
    Normalize a URL: convert http→https, extract #fragment, resolve redirects.
    Returns (base_url, fragment) where fragment includes '#' prefix or is ''.
    """
    if not url or not str(url).strip():
        return "", ""
    if url.startswith('http://'):
        url = 'https://' + url[7:]
    fragment = ''
    if '#' in url:
        url, fragment = url.split('#', 1)
        fragment = '#' + fragment
    resolved = resolve_url_redirects(url)
    if resolved != url:
        if '#' in resolved:
            resolved, redirect_fragment = resolved.split('#', 1)
            if not fragment:
                fragment = '#' + redirect_fragment
        url = resolved
    return url, fragment

# description: build the deep-link text payload
def _build_deep_link_text(page_content: str, metadata: dict, title: str = '', url: str = '') -> str:
    """
    Build the deep-link text payload based on DEEP_LINK_HIGHLIGHT_MODE.
    Returns the text that will be URL-encoded into ?diplo-deep-link-text=...
    """
    # get best sentence from the metadata it is formed in the sentence_first.py file
    # first from v4 client, than with TEI reranker for best sentence in the group (url, section (sentence))
    best_sent = metadata.get('_best_sentence', '')
    # get matched sentences from the metadata it is formed in the sentence_first.py file
    matched_sents = metadata.get('_matched_sentences', [])
    # get the deep link highlight mode from the configuration
    _dl_mode = get_param('deep_link_highlight_mode', DEEP_LINK_HIGHLIGHT_MODE)
    _sh_mode = get_param('sentence_highlight_mode', SENTENCE_HIGHLIGHT_MODE)
    if _dl_mode == 'sentence' and best_sent:
        if _sh_mode == 'multi' and matched_sents:
            sent_payload = '|||SENT|||'.join(matched_sents)
        else:
            sent_payload = best_sent
        return page_content + '\n===SENTENCE===\n' + sent_payload
    else:
        text = ''
        if title:
            text += title + '\n'
        if url:
            text += url + '\n\n'
        text += page_content
        return text

# this is helper function that is used to check if the url is a pdf url
def _is_pdf_url(url: str) -> bool:
    """True when the path looks like a PDF (suffix, mid-path, or /pdfs/)."""
    if not url:
        return False
    path = url.split('?')[0].split('#')[0].lower().rstrip('/')
    if path.endswith('.pdf') or '.pdf/' in path + '/':
        return True
    if '/pdfs/' in path or path.endswith('.pdf%20') or '.pdf%' in path:
        return True
    return '.pdf' in path


def _build_doc_viewer_link(
    metadata: dict,
    page_content: str,
    title: str = '',
) -> str:
    """Deep link into our own viewer for documents that have no source_url."""
    if not DOC_VIEWER_BASE_URL:
        return ''
    doc_hash = str(
        (metadata or {}).get('parent_document_hash')
        or (metadata or {}).get('document_hash')
        or ''
    ).strip()
    if not doc_hash:
        return ''

    base = f"{DOC_VIEWER_BASE_URL.rstrip('/')}/api/doc/{doc_hash}"
    deep_link_text = _build_deep_link_text(page_content, metadata, title, '')
    if not deep_link_text:
        return base
    return f"{base}#diplo-deep-link-text={quote(deep_link_text, safe='')}"


def _via_proxy_url(pdf_url: str) -> str:
    """Route a raw PDF URL through chatbot-via (no hash / query)."""
    proxy = (PDF_PROXY_BASE_URL or '').rstrip('/')
    raw = (pdf_url or '').strip()
    if not proxy or not raw:
        return raw
    if raw.startswith(proxy + '/'):
        return raw
    return f"{proxy}/{raw}"


def _host_runs_highlight_script(url: str) -> bool:
    """True when the target host serves the Diplo highlight script."""
    host = (urlparse(url).hostname or '').lower()
    if not host:
        return False
    return any(
        host == suffix or host.endswith('.' + suffix)
        for suffix in HIGHLIGHT_SCRIPT_HOSTS
    )


_FRAGMENT_EDGE_WORDS = 8


def _encode_fragment_part(text: str) -> str:
    """Percent-encode for ``#:~:text=``; ``-`` and ``,`` are syntax there."""
    return quote(text, safe='').replace('-', '%2D')


def _text_fragment(page_content: str, metadata: dict) -> str:
    """Browser-native ``:~:text=`` payload for pages without our highlight script."""
    snippet = (metadata.get('_best_sentence') or '').strip()
    if not snippet:
        matched = metadata.get('_matched_sentences') or []
        snippet = (matched[0] if matched else (page_content or '')).strip()
    snippet = re.sub(r'\s+', ' ', snippet).strip()
    if len(snippet) < 12:
        return ''

    words = snippet.split(' ')
    if len(words) <= 2 * _FRAGMENT_EDGE_WORDS:
        return f":~:text={_encode_fragment_part(snippet)}"
    # textStart,textEnd keeps the URL short and survives mid-passage edits.
    start = ' '.join(words[:_FRAGMENT_EDGE_WORDS])
    end = ' '.join(words[-_FRAGMENT_EDGE_WORDS:])
    return f":~:text={_encode_fragment_part(start)},{_encode_fragment_part(end)}"


def _build_native_fragment_link(
    url: str,
    fragment: str,
    page_content: str,
    metadata: dict,
) -> str:
    """Third-party page URL carrying a browser-native text fragment."""
    text_fragment = _text_fragment(page_content, metadata)
    if not text_fragment:
        return url + fragment
    # Syntax is ``#<element>:~:text=``, so an existing anchor stays intact.
    return f"{url}{fragment}{text_fragment}" if fragment else f"{url}#{text_fragment}"


def _source_card_deep_link(
    source: dict,
    query: str = None,
    deep_link_service: DeepLinkService = None,
    preferred: str | None = None,
) -> str | None:
    """Clickable href for a Sources card — same chatbot-via URL as citations.

    WordPress uses ``deep_link_url`` as-is. If it is missing it appends ``/``
    to ``url`` and adds ``?diplo-deep-link-text=``, which breaks PDFs.
    """
    if preferred and (
        not _is_pdf_url(preferred)
        or (PDF_PROXY_BASE_URL and PDF_PROXY_BASE_URL.rstrip('/') in preferred)
    ):
        return preferred

    href = build_deep_link_url(
        source, query=query, deep_link_service=deep_link_service
    )
    if href:
        return href
    raw = (source.get('url') or source.get('link') or '').strip()
    if raw and _is_pdf_url(raw) and PDF_PROXY_BASE_URL:
        return _via_proxy_url(raw)
    return raw or None


# this is helper function that is used to build the deep link url for a pdf url
def _build_pdf_deep_link(
    pdf_url: str,
    page_content: str,
    metadata: dict,
    title: str,
    query: str = None,
    deep_link_service: DeepLinkService = None,
) -> str:
    """Build a deep-link URL for a PDF routed through the chatbot-via proxy.

    Uses hash fragments instead of query params because pdfjs-init.js reads
    highlight parameters from location.hash.
    """
    proxy = PDF_PROXY_BASE_URL.rstrip('/')
    base = f"{proxy}/{pdf_url}"

    deep_link_text = _build_deep_link_text(page_content, metadata, title, pdf_url)
    matched_sents = metadata.get('_matched_sentences', [])

    _use_short = get_param('use_short_deep_links', USE_SHORT_DEEP_LINKS)
    _sh_mode = get_param('sentence_highlight_mode', SENTENCE_HIGHLIGHT_MODE)

    hash_parts = []

    if _use_short and _sh_mode == 'multi' and matched_sents and deep_link_service:
        try:
            dl_metadata = {"url": pdf_url, "title": title}
            if query:
                dl_metadata["query"] = query
            dl_id = deep_link_service.create_deep_link(
                section_text=page_content,
                sentences=matched_sents,
                metadata=dl_metadata,
            )
            print(f"DEBUG: PDF short link {dl_id} ({len(matched_sents)} sents)", flush=True)
            hash_parts.append(f"diplo-hl-id={dl_id}")
            if DEEP_LINK_API_URL:
                hash_parts.append(f"diplo-hl-api={quote(DEEP_LINK_API_URL, safe='')}")
        except Exception as e:
            logger.warning(f"PDF short link failed, fallback: {e}")
            sent_payload = '|||SENT|||'.join(matched_sents[:2])
            fallback_text = page_content + '\n===SENTENCE===\n' + sent_payload
            hash_parts.append(f"diplo-deep-link-text={quote(fallback_text, safe='')}")
    elif deep_link_text:
        hash_parts.append(f"diplo-deep-link-text={quote(deep_link_text, safe='')}")

    if hash_parts:
        return f"{base}#{'&'.join(hash_parts)}"
    return base

# this is helper function that is used to build the deep link url for a document
def build_deep_link_url(
    doc_or_dict,
    query: str = None,
    deep_link_service: DeepLinkService = None
) -> str:
    """
    Build a deep-link URL for a document (Document object or plain dict).

    Consolidates the 4 previous copies into one canonical implementation:
      - Unified metadata accessor via get_metadata_field()
      - HTTP→HTTPS + fragment extraction + redirect resolution
      - Sentence / paragraph text building
      - Redis shortening (when deep_link_service is provided and multi-mode)
      - Fallback: 2-sentence cap on Redis error, else URL param encoding
    """
    if hasattr(doc_or_dict, 'metadata'):
        metadata = doc_or_dict.metadata
        page_content = doc_or_dict.page_content
    else:
        metadata = doc_or_dict
        page_content = metadata.get('text', '')

    url = get_metadata_field(metadata, 'url')
    title = get_metadata_field(metadata, 'title')

    # Normalize URL
    url, fragment = _normalize_url_with_redirect(url)

    # Uploaded files have no public URL — serve them from our own viewer.
    if not url:
        return _build_doc_viewer_link(metadata, page_content, title)

    # PDF sources: route through chatbot-via proxy with hash-based params
    if PDF_PROXY_BASE_URL and _is_pdf_url(url):
        return _build_pdf_deep_link(url, page_content, metadata, title, query, deep_link_service)

    # Third-party pages: our highlight script is not there to read
    # ?diplo-deep-link-text=, so let the browser do the highlighting.
    if not _host_runs_highlight_script(url):
        return _build_native_fragment_link(url, fragment, page_content, metadata)

    # Build text
    matched_sents = metadata.get('_matched_sentences', [])
    deep_link_text = _build_deep_link_text(page_content, metadata, title, url)

    _use_short = get_param('use_short_deep_links', USE_SHORT_DEEP_LINKS)
    _sh_mode = get_param('sentence_highlight_mode', SENTENCE_HIGHLIGHT_MODE)
    if _use_short and _sh_mode == 'multi' and matched_sents and url and deep_link_service:
        try:
            dl_metadata = {"url": url, "title": title}
            if query:
                dl_metadata["query"] = query
            dl_id = deep_link_service.create_deep_link(
                section_text=page_content,
                sentences=matched_sents,
                metadata=dl_metadata
            )
            print(f"DEBUG: Short link {dl_id} ({len(matched_sents)} sents)", flush=True)
            hl_api_param = f"&diplo-hl-api={quote_plus(DEEP_LINK_API_URL)}" if DEEP_LINK_API_URL else ""
            return f"{url}?diplo-hl-id={dl_id}{hl_api_param}{fragment}"
        except Exception as e:
            logger.warning(f"Short link failed, fallback: {e}")
            sent_payload = '|||SENT|||'.join(matched_sents[:2])
            fallback_text = page_content + '\n===SENTENCE===\n' + sent_payload
            return f"{url}?diplo-deep-link-text={quote_plus(fallback_text)}{fragment}"

    # URL param fallback
    if url and deep_link_text:
        return f"{url}?diplo-deep-link-text={quote_plus(deep_link_text)}{fragment}"
    return url + fragment


def build_deep_link_urls_batch(
    docs,
    query: str = None,
    deep_link_service: DeepLinkService = None
) -> dict[int, str]:
    """
    Build deep link URLs for multiple documents using batch Redis creation.
    Returns {1-based_index: deep_link_url} dict.
    """
    result = {}
    batch_items = []

    for i, doc in enumerate(docs):
        if hasattr(doc, 'metadata'):
            metadata = doc.metadata
            page_content = doc.page_content
        else:
            metadata = doc
            page_content = metadata.get('text', '')

        url = get_metadata_field(metadata, 'url')
        title = get_metadata_field(metadata, 'title')
        url, fragment = _normalize_url_with_redirect(url)

        # Uploaded files have no public URL — serve them from our own viewer.
        if not url:
            viewer = _build_doc_viewer_link(metadata, page_content, title)
            if viewer:
                result[i + 1] = viewer
            continue

        # PDF sources: handle individually via chatbot-via proxy
        if PDF_PROXY_BASE_URL and _is_pdf_url(url):
            result[i + 1] = _build_pdf_deep_link(url, page_content, metadata, title, query, deep_link_service)
            continue

        # Third-party pages: browser-native highlight, see build_deep_link_url().
        if not _host_runs_highlight_script(url):
            result[i + 1] = _build_native_fragment_link(url, fragment, page_content, metadata)
            continue

        matched_sents = metadata.get('_matched_sentences', [])
        deep_link_text = _build_deep_link_text(page_content, metadata, title, url)

        _use_short = get_param('use_short_deep_links', USE_SHORT_DEEP_LINKS)
        _sh_mode = get_param('sentence_highlight_mode', SENTENCE_HIGHLIGHT_MODE)
        if _use_short and _sh_mode == 'multi' and matched_sents and url and deep_link_service:
            dl_metadata = {"url": url, "title": title}
            if query:
                dl_metadata["query"] = query
            batch_items.append({
                "index": i + 1,
                "section_text": page_content,
                "sentences": matched_sents,
                "metadata": dl_metadata,
                "url": url,
                "fragment": fragment,
                "deep_link_text": deep_link_text,
            })
        else:
            if url and deep_link_text:
                result[i + 1] = f"{url}?diplo-deep-link-text={quote_plus(deep_link_text)}{fragment}"
            else:
                result[i + 1] = url + fragment

    if batch_items and deep_link_service:
        links_input = [
            {"section_text": bi["section_text"], "sentences": bi["sentences"], "metadata": bi["metadata"]}
            for bi in batch_items
        ]
        try:
            ids = deep_link_service.create_deep_links_batch(links_input)
            hl_api_param = f"&diplo-hl-api={quote_plus(DEEP_LINK_API_URL)}" if DEEP_LINK_API_URL else ""
            for bi, dl_id in zip(batch_items, ids):
                if dl_id:
                    result[bi["index"]] = f"{bi['url']}?diplo-hl-id={dl_id}{hl_api_param}{bi['fragment']}"
                else:
                    sent_payload = '|||SENT|||'.join(bi["sentences"][:2])
                    fallback_text = bi["section_text"] + '\n===SENTENCE===\n' + sent_payload
                    result[bi["index"]] = f"{bi['url']}?diplo-deep-link-text={quote_plus(fallback_text)}{bi['fragment']}"
            print(f"DEBUG: Batch created {len(ids)} short links", flush=True)
        except Exception as e:
            logger.warning(f"Batch short link creation failed, fallback: {e}")
            for bi in batch_items:
                sent_payload = '|||SENT|||'.join(bi["sentences"][:2])
                fallback_text = bi["section_text"] + '\n===SENTENCE===\n' + sent_payload
                result[bi["index"]] = f"{bi['url']}?diplo-deep-link-text={quote_plus(fallback_text)}{bi['fragment']}"

    return result


def parse_citations(answer: str) -> list[int]:
    """
    Parse citation indices from answer text.
    Looks for patterns like [1], [2], [1][3], etc.
    Returns list of unique 0-based indices.
    """
    # Find all [N] patterns
    citations = re.findall(r'\[(\d+)\]', answer)
    # Convert to 0-based indices and deduplicate
    indices = list(set(int(c) - 1 for c in citations if c.isdigit()))
    return sorted(indices)


async def generate_related_questions_from_sources(
    sources: list,
    original_question: str,
    llm,
    *,
    langfuse_session_id: str | None = None,
    langfuse_user_id: str | None = None,
) -> list[str]:
    """
    Generate related questions from source documents using LLM.
    Returns a list of 3-5 related questions that users might want to explore.
    
    Model selection is configured via .env:
    - RELATED_QUESTIONS_PROVIDER = 'openai', 'local', or 'deepseek'
    - For 'openai': uses RELATED_QUESTIONS_OPENAI_MODEL (default: gpt-4o-mini)
    - For 'local': uses RELATED_QUESTIONS_LOCAL_URL, RELATED_QUESTIONS_LOCAL_MODEL, RELATED_QUESTIONS_LOCAL_KEY
    - For 'deepseek': uses RELATED_QUESTIONS_DEEPSEEK_MODEL + DEEPSEEK_API_KEY / DEEPSEEK_API_URL
    """
    from langchain_openai import ChatOpenAI
    from app.core.config import (
        OPENAI_KEY, OPENAI_ORGANIZATION,
        DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
        RELATED_QUESTIONS_PROVIDER,
        RELATED_QUESTIONS_LOCAL_URL, RELATED_QUESTIONS_LOCAL_MODEL, RELATED_QUESTIONS_LOCAL_KEY,
        RELATED_QUESTIONS_OPENAI_MODEL, RELATED_QUESTIONS_DEEPSEEK_MODEL,
    )
    
    if not sources:
        return []
    
    # Extract key content from sources (limit to avoid token overflow)
    source_summaries = []
    for i, doc in enumerate(sources[:5]):  # Max 5 sources
        title = doc.metadata.get('title', doc.metadata.get('name', ''))
        content = doc.page_content[:500] if doc.page_content else ''
        if title or content:
            source_summaries.append(f"Source {i+1}: {title}\n{content}")
    
    if not source_summaries:
        return []
    
    sources_text = "\n\n".join(source_summaries)
    
    prompt = f"""Based on the following sources that were used to answer the question "{original_question}", generate 3-5 follow-up questions that a user might want to explore next.

Sources:
{sources_text}

Requirements:
- Questions should be related to but different from the original question
- Questions should be answerable from the knowledge in these sources or related topics
- Keep questions concise (under 60 characters each)
- Return ONLY the questions, one per line, no numbering or bullets

Questions:"""

    try:
        from app.core.config import (
            LLM_FAILSAFE_ENABLED, FAILSAFE_LLM_PROVIDER, FAILSAFE_LLM_MODEL,
            LLM_MODEL_PROVIDER, LLM_MODEL_NAME
        )
        
        # Select model based on configuration
        provider = RELATED_QUESTIONS_PROVIDER.lower().strip()
        
        if provider == 'local':
            print(f"DEBUG: Using LOCAL model for related questions: {RELATED_QUESTIONS_LOCAL_MODEL}", flush=True)
            questions_llm = ChatOpenAI(
                model_name=RELATED_QUESTIONS_LOCAL_MODEL,
                temperature=0.7,
                api_key=RELATED_QUESTIONS_LOCAL_KEY,
                base_url=RELATED_QUESTIONS_LOCAL_URL
            )
        elif provider == 'deepseek':
            print(f"DEBUG: Using DEEPSEEK model for related questions: {RELATED_QUESTIONS_DEEPSEEK_MODEL}", flush=True)
            questions_llm = ChatOpenAI(
                model_name=RELATED_QUESTIONS_DEEPSEEK_MODEL,
                temperature=0.7,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_API_URL,
            )
        else:
            print(f"DEBUG: Using OPENAI model for related questions: {RELATED_QUESTIONS_OPENAI_MODEL}", flush=True)
            questions_llm = ChatOpenAI(
                model_name=RELATED_QUESTIONS_OPENAI_MODEL,
                temperature=0.7,
                api_key=OPENAI_KEY,
                organization=OPENAI_ORGANIZATION
            )
        
        if LLM_FAILSAFE_ENABLED and FAILSAFE_LLM_PROVIDER and FAILSAFE_LLM_PROVIDER != provider:
            if FAILSAFE_LLM_PROVIDER == 'local':
                fallback_rq = ChatOpenAI(
                    model_name=RELATED_QUESTIONS_LOCAL_MODEL,
                    temperature=0.7,
                    api_key=RELATED_QUESTIONS_LOCAL_KEY,
                    base_url=RELATED_QUESTIONS_LOCAL_URL)
            elif FAILSAFE_LLM_PROVIDER == 'deepseek':
                fallback_rq = ChatOpenAI(
                    model_name=FAILSAFE_LLM_MODEL or RELATED_QUESTIONS_DEEPSEEK_MODEL,
                    temperature=0.7,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_API_URL)
            else:
                fallback_rq = ChatOpenAI(
                    model_name=FAILSAFE_LLM_MODEL or RELATED_QUESTIONS_OPENAI_MODEL,
                    temperature=0.7,
                    api_key=OPENAI_KEY, organization=OPENAI_ORGANIZATION)
            questions_llm = questions_llm.with_fallbacks([fallback_rq])
        
        response = await questions_llm.ainvoke(
            [{'role': 'human', 'content': prompt}],
            config=langfuse_invoke_config(
                thread_id=langfuse_session_id,
                user_id=langfuse_user_id,
                run_name="related_questions",
            ),
        )
        questions = [q.strip() for q in response.content.strip().split('\n') if q.strip() and '?' in q]
        # Clean up any numbering or bullets
        questions = [re.sub(r'^[\d\.\-\*\)]+\s*', '', q).strip() for q in questions]
        return questions[:5]  # Max 5 questions
    except Exception as e:
        print(f"WARNING: Failed to generate related questions: {e}", flush=True)
        return []


def get_metadata_field(metadata: dict, field: str) -> str:
    """
    Get metadata field with fallback for DiploParagraph / oneweaviate.
    Maps: 'link' → 'url', 'name' → 'title'; recovers URL from encoded titles.
    Returns empty string if field not found.
    """
    metadata = metadata or {}
    if field == 'url':
        from app.core.weaviate_props import doc_url
        recovered = doc_url(metadata)
        if recovered:
            return recovered
        url = (metadata.get('url') or '').strip()
        if url:
            return url
        return (metadata.get('link') or '').strip()
    elif field == 'title':
        from app.core.weaviate_props import doc_title
        title = doc_title(metadata)
        if title and title != 'Untitled':
            return title
        title = (metadata.get('title') or '').strip()
        if title:
            return title
        return (metadata.get('name') or '').strip()
    else:
        return metadata.get(field) or ''

# context: this class is called from chat_route.py than gives the methods for use
# description: ChatService class - this is the main class that handles the chat requests orchestrates the chatbot and the retriever
class ChatService:
    def __init__(self, use_sentence_header: bool = False):
        # use_sentence_header=True → A/B graph (title+sentence hybrid) for /api/chat-sh
        singleton = Singleton()
        if use_sentence_header:
            self.diplomacy_bot = singleton.diplomacy_bot_sh
            self.label_retriever = singleton.label_retriever_sh
        else:
            self.diplomacy_bot = singleton.diplomacy_bot
            self.label_retriever = singleton.label_retriever
        self.conversation_history = singleton.conversation_history
        self.chat_repository = ChatRepository()
        self.response_repository = ResponseRepository()
        self.resource_filter = singleton.resource_filter

    # context: called by handle_chat_request_WS to send the progress to the frontend
    # description: connect to websocket -> get info on progress -> string and payload
    async def _send_progress(
        self,
        websocket: WebSocket,
        phase: str,
        text: str,
        **extra,
    ):
        """WS progress chip (rag_progress custom events from the graph)."""
        payload = {"status": "progress", "phase": phase, "text": text}
        if extra:
            payload.update(extra)
        await Singleton().websocket_manager.send_message(payload, websocket)

# description: handle the websocket connection 
# parameters:
# - conversation_id: str - the conversation id - created in conversation_route.py (conversation id generator)
# - data: ChatRouteResponse - the response from the chat service
# - websocket: WebSocket - the websocket connection
# returns:
# - None
    async def handle_chat_request_WS(self, conversation_id: str, data: ChatRouteResponse, websocket: WebSocket):
        # set the user type
        user_type_str = data.user_type.value if hasattr(data.user_type, 'value') else str(data.user_type)
        # from frontend we get the data -> the type is called ChatRouteResponse - this is a mistake - it should be ChatRouteRequest
        original_message = data.message

        # takes configuration sent from WP (frontend) and activates it for this request (chatbot interaction)
        # config is stored in data inside the retrieval_config key
        if hasattr(data, 'retrieval_config') and data.retrieval_config:
            overrides = {k: v for k, v in data.retrieval_config.dict().items() if v is not None}
            set_retrieval_overrides(overrides)
            print(f"DEBUG chat_service: Retrieval overrides set: {overrides}", flush=True)
        else:
            overrides = {}
            clear_retrieval_overrides()
        # if we setup the custom filters -> here we override the metadata
        filter_metadata = _filter_langfuse_metadata(overrides)
        config = build_langfuse_run_config(
            thread_id=conversation_id,
            user_id=data.user_ip,
            run_name="chat_ws",
            tags=[user_type_str],
            extra_metadata=filter_metadata, # here we override the metadata
        )
        thread_config = build_thread_config(conversation_id)
        
        # Set user profile for label-based scoring
        # debug for user_type
        print(f"DEBUG chat_service: Setting user_type to '{user_type_str}'", flush=True)
        self.label_retriever.set_user_type(user_type_str)
        # inside var put role and conversation history
        memory = [{'role': c['role'], 'content': c['content']} for c in  self.conversation_history[conversation_id]]
        # this checks if WP sends a custom system prompt - and if so, it uses it - otherwise it uses the default system prompt
        system_prompt = _resolve_system_prompt(data)
        
        # those 2 from above put inside var history (user_type + system prompt)
        history = repack_chat_history(memory, system_prompt)
       
        # update state with config and history
        state_update = {'messages': [SystemMessage(content=system_prompt)]}
        # aupdate_state is a method that updates the state of the chatbot - his is in a way adapter for updating the state of the chatbot
        # parameters:
        # - thread_config: id of the conversation for tracking
        # - state_update: what are we sending to other functions in the chatbot
        await self.diplomacy_bot.aupdate_state(thread_config, state_update)
        # start time of the chatbot response
        start_time = datetime.now()
 
        # answer is empty string - will be filled with the chatbot response
        answer = ''
        # retrieved_docs is empty list - will be filled with the retrieved documents
        # this is good for DEBUGGING
        retrieved_docs = []
        # related_questions is empty list - will be filled with the related questions
        related_questions = []
        
        # Live renumbering: track citations as they appear in stream
        seen_citations = {}  # {orig_num: new_num}
        next_citation_num = [1]  # Use list to allow modification in nested function
        stream_buffer = '' # string that will be sent
        doc_urls = {}  # {orig_num: deep_link_url} - populated when retriever returns
        sent_citation_urls = set()  # Track which citation URLs already sent
        
        # deep link service is used to build the deep link urls for the citations
        dl_service = DeepLinkService() if get_param('use_short_deep_links', USE_SHORT_DEEP_LINKS) else None
        
        # this is the main loop that streams the chatbot response
        # it is streamed in chunks - via websocket endpoint
        stream_input = {'messages': data.message}
        # this are open for testing - we will remove them later
        early_sources_sent = False
        early_graph_sent = False
        # when connected to websocket -> send retrieving as phase and Find relevant sources... as text
        await self._send_progress(
            websocket,
            "retrieving",
            "Finding relevant sources…",
        )
        # for every event that astream_events gives us -> we process it
        async for event in self.diplomacy_bot.astream_events(stream_input, config=config, version='v2'):
            # put ['event'] into event_type var - ['event'] all inputs that astream_events listens to
            # ['event'] is langchain dictionary for astream_events library
            event_type = event['event']
            # if there is a custom event -> we process it - this is from astream_events library
            if event_type == "on_custom_event":
            # get event name

            # if it is rag_progress -> we process it
                if event.get("name") == "rag_progress":
                    # get the data from the event
                    payload = event.get("data") or {}
                    phase = payload.get("phase") or "working"
                    text = payload.get("text") or "Working…"
                    extra = {
                        k: v
                        for k, v in payload.items()
                        if k not in ("phase", "text")
                    }
                    await self._send_progress(websocket, phase, text, **extra)
                    # Stream RAG source cards before the answer (same status=sources as final)
                    if phase == "sources_ready" and not early_sources_sent:
                        items = extra.get("items") or payload.get("items") or []
                        # on_retriever_end fires before this event and consumes
                        # _last_reranked_docs, so fall back to what it captured.
                        docs = (
                            getattr(self.label_retriever, "_last_reranked_docs", None)
                            or retrieved_docs
                            or []
                        )
                        if docs:
                            retrieved_docs = list(docs)
                            related_questions = [
                                doc.metadata.get(
                                    "questions_this_excerpt_can_answer", ""
                                )
                                for doc in retrieved_docs
                            ]
                            doc_urls = build_deep_link_urls_batch(
                                retrieved_docs,
                                query=data.message,
                                deep_link_service=dl_service,
                            )
                            sources = self.__repack_source_documents(
                                retrieved_docs, doc_urls
                            )
                            sources_dicts = [
                                s.to_dict()
                                for s in sources
                                if s is not None
                            ]
                            sources_dicts = [
                                sd for sd in sources_dicts if sd is not None
                            ]
                        elif items:
                            # Fallback: cards from progress payload — still attach
                            # chatbot-via deep_link_url so WP does not mangle PDFs.
                            sources_dicts = []
                            for it in items:
                                if not isinstance(it, dict):
                                    continue
                                card = {
                                    "title": it.get("title") or "Untitled",
                                    "url": it.get("url") or "",
                                    "text": it.get("text") or "",
                                    "date": it.get("date") or "Unknown date",
                                }
                                card["deep_link_url"] = _source_card_deep_link(
                                    card,
                                    query=data.message,
                                    deep_link_service=dl_service,
                                )
                                sources_dicts.append(card)
                        else:
                            sources_dicts = []
                        if sources_dicts:
                            await Singleton().websocket_manager.send_message(
                                {
                                    "status": "sources",
                                    "text": sources_dicts,
                                },
                                websocket,
                            )
                            early_sources_sent = True
                            print(
                                f"DEBUG WS early sources: {len(sources_dicts)} "
                                f"(pre-answer RAG set, docs={len(docs)})",
                                flush=True,
                            )
                            for _i, _sd in enumerate(sources_dicts):
                                _dl = _sd.get("deep_link_url") or ""
                                print(
                                    f"DEBUG WS early source[{_i}]: "
                                    f"via={_dl.startswith(PDF_PROXY_BASE_URL or '#')} "
                                    f"url={(_sd.get('url') or '')[:90]!r} "
                                    f"dl={_dl[:90]!r}",
                                    flush=True,
                                )
                    if phase == "graph_ready" and not early_graph_sent:
                        gdata = (
                            extra.get("graph_data")
                            or payload.get("graph_data")
                        )
                        if gdata and (
                            gdata.get("subgraph_url")
                            or gdata.get("nodes")
                        ):
                            await Singleton().websocket_manager.send_message(
                                {"status": "graph_data", "text": gdata},
                                websocket,
                            )
                            early_graph_sent = True
                            print(
                                "DEBUG WS early graph_data: "
                                f"nodes={len(gdata.get('nodes') or [])} "
                                f"url={gdata.get('subgraph_url')}",
                                flush=True,
                            )
            if event_type =='on_chat_model_stream':
                content = event['data']['chunk'].content
                if content:
                    if isinstance(content, list) and 'text' in content[0].keys():
                        text = content[0]['text']
                    else:
                        text = content
                    
                    # Live citation renumbering during streaming
                    # Buffer text to handle citations split across chunks
                    stream_buffer += text
                    
                    # Find safe point to send (before any incomplete citation)
                    last_bracket = stream_buffer.rfind('[')
                    if last_bracket == -1:
                        # No bracket, safe to send all
                        safe_text = stream_buffer
                        stream_buffer = ''
                    else:
                        # Check if there's a ']' after the last '['
                        after_bracket = stream_buffer[last_bracket:]
                        if ']' in after_bracket:
                            # Complete citation, safe to send all
                            safe_text = stream_buffer
                            stream_buffer = ''
                        else:
                            # Incomplete citation, send up to bracket
                            safe_text = stream_buffer[:last_bracket]
                            stream_buffer = stream_buffer[last_bracket:]
                    
                    if safe_text:
                        # Normalize fancy citation formats to [N]
                        safe_text = re.sub(r'[\u3010\u3014\uFF3B\[]\s*(\d+)\s*[\u3011\u3015\uFF3D\]]', r'[\1]', safe_text)
                        safe_text = re.sub(r'\d+†L\d+(?:-L?\d+)?', '', safe_text)
                        # Find and renumber citations, sending URLs immediately
                        def renumber_and_track(match):
                            orig = int(match.group(1))
                            if orig not in seen_citations:
                                seen_citations[orig] = next_citation_num[0]
                                next_citation_num[0] += 1
                            return f'[{seen_citations[orig]}]'
                        
                        processed_text = re.sub(r'\[(\d+)\]', renumber_and_track, safe_text)
                        # Remove commas/spaces between consecutive citations: [1], [2] or [1] [2] -> [1][2]
                        processed_text = re.sub(r'\][\s,]+\[', '][', processed_text)
                        
                        # Send answer text FIRST
                        message = {
                            "status": "answer",
                            "text": processed_text
                        }
                        await Singleton().websocket_manager.send_message(message, websocket)
                        answer += processed_text
                        
                        # THEN send citation URLs for any new citations found
                        for orig_num, new_num in seen_citations.items():
                            if new_num not in sent_citation_urls and orig_num in doc_urls:
                                # Send this citation URL immediately after text
                                await Singleton().websocket_manager.send_message({
                                    "status": "citation_url",  # Single URL, not plural
                                    "num": new_num,
                                    "url": doc_urls[orig_num]
                                }, websocket)
                                sent_citation_urls.add(new_num)
                    
            elif event_type == 'on_retriever_end':
                # Hvati samo PRVI retriever rezultat, ignoriši ostale (LLM može halucinirati višestruke pozive)
                if not retrieved_docs:
                    # Prefer post-reranking docs stored on the retriever
                    # (on_retriever_end fires pre-reranking, causing citation mismatch)
                    reranked = getattr(self.label_retriever, '_last_reranked_docs', None)
                    if reranked:
                        retrieved_docs = reranked
                        self.label_retriever._last_reranked_docs = None  # Clear for next request
                    else:
                        retrieved_docs = event['data']['output']
                    related_questions = [doc.metadata.get('questions_this_excerpt_can_answer', '') for doc in retrieved_docs]
                    # Pre-compute deep link URLs for all documents (batch Redis pipeline)
                    doc_urls = build_deep_link_urls_batch(retrieved_docs, query=data.message, deep_link_service=dl_service)
        
        # Send any remaining buffered text
        if stream_buffer:
            # Normalize fancy citation formats to [N]
            stream_buffer = re.sub(r'[\u3010\u3014\uFF3B\[]\s*(\d+)\s*[\u3011\u3015\uFF3D\]]', r'[\1]', stream_buffer)
            stream_buffer = re.sub(r'\d+†L\d+(?:-L?\d+)?', '', stream_buffer)
            def renumber_and_track(match):
                orig = int(match.group(1))
                if orig not in seen_citations:
                    seen_citations[orig] = next_citation_num[0]
                    next_citation_num[0] += 1
                return f'[{seen_citations[orig]}]'
            
            processed_text = re.sub(r'\[(\d+)\]', renumber_and_track, stream_buffer)
            # Remove commas/spaces between consecutive citations: [1], [2] or [1] [2] -> [1][2]
            processed_text = re.sub(r'\][\s,]+\[', '][', processed_text)
            
            # Send text FIRST
            message = {
                "status": "answer",
                "text": processed_text
            }
            await Singleton().websocket_manager.send_message(message, websocket)
            answer += processed_text
            
            # THEN send any remaining citation URLs
            for orig_num, new_num in seen_citations.items():
                if new_num not in sent_citation_urls and orig_num in doc_urls:
                    await Singleton().websocket_manager.send_message({
                        "status": "citation_url",
                        "num": new_num,
                        "url": doc_urls[orig_num]
                    }, websocket)
                    sent_citation_urls.add(new_num)
        
        _lf_handler = get_langfuse_callback_handler()
        patch_langfuse_trace_metadata(
            filter_metadata,
            trace_id=getattr(_lf_handler, "last_trace_id", None) if _lf_handler else None,
        )
        
        # Fallback: if no on_retriever_end fired (e.g. cache hit skipped retrieval),
        # check _last_reranked_docs stored by direct_retrieval_node.
        if not retrieved_docs:
            reranked = getattr(self.label_retriever, '_last_reranked_docs', None)
            if reranked:
                retrieved_docs = reranked
                self.label_retriever._last_reranked_docs = None
                related_questions = [doc.metadata.get('questions_this_excerpt_can_answer', '') for doc in retrieved_docs]
                doc_urls = build_deep_link_urls_batch(retrieved_docs, query=data.message, deep_link_service=dl_service)
                print(f"DEBUG: Recovered {len(retrieved_docs)} docs from _last_reranked_docs (cache hit path)", flush=True)
        
        # Filter responses
        # Pass seen_citations from live renumbering - answer already has sequential citations
        relevant_source_documents, related_questions, citation_urls, renumber_map = self.filter_related_sources(
            retrieved_docs, data.message, answer, seen_citations,
            deep_link_service=dl_service,
            langfuse_session_id=conversation_id,
            langfuse_user_id=data.user_ip,
        )
        
        # Generate related questions from sources if empty
        if not related_questions and relevant_source_documents:
            try:
                import time as _time
                t_rq = _time.time()
                related_questions = await generate_related_questions_from_sources(
                    relevant_source_documents,
                    data.message,
                    self.resource_filter,
                    langfuse_session_id=conversation_id,
                    langfuse_user_id=data.user_ip,
                )
                print(f"TIMING related_questions: {_time.time()-t_rq:.3f}s", flush=True)
            except Exception as e:
                print(f"WARNING: Failed to generate related questions: {e}", flush=True)
                related_questions = []
        
        end_time = datetime.now()

        response_time = (end_time - start_time).total_seconds()

        # Pass citation_urls so each source gets a pre-built deep_link_url
        # Do NOT overwrite doc.metadata['url'] - that causes double deep link in PHP
        sources = self.__repack_source_documents(relevant_source_documents, citation_urls)
        
        # Converting to dict and than to JSON - SEND SOURCES FIRST before MongoDB operations
        sources_dicts = [source.to_dict() for source in sources if source is not None]
        sources_dicts = [sd for sd in sources_dicts if sd is not None]
        
        # Debug: verify deep_link_url is in sources
        for idx, sd in enumerate(sources_dicts):
            has_dl = 'deep_link_url' in sd and bool(sd.get('deep_link_url'))
            has_sentence = '===SENTENCE===' in (sd.get('deep_link_url', '') or '') or '%3D%3D%3DSENTENCE%3D%3D%3D' in (sd.get('deep_link_url', '') or '')
            print(f"DEBUG WS source[{idx}]: has_deep_link_url={has_dl}, has_SENTENCE={has_sentence}, keys={list(sd.keys())}", flush=True)
        
        # Send sources and related data BEFORE MongoDB saves (so even if MongoDB fails, user gets sources)
        import json as _json
        if early_sources_sent:
            # Already streamed the RAG source set before the answer; skip duplicate cards
            print(
                "DEBUG WS sources: skipped final send (early_sources_sent)",
                flush=True,
            )
        else:
            _sources_json = _json.dumps({"status": "sources", "text": sources_dicts})
            print(f"DEBUG WS sources JSON size: {len(_sources_json)} bytes, first 300 chars: {_sources_json[:300]}", flush=True)
            print(f"DEBUG WS sources JSON last 300 chars: {_sources_json[-300:]}", flush=True)
            await Singleton().websocket_manager.send_message({"status": "sources", "text": sources_dicts}, websocket)

        try:
            from app.ai.ai_services.graph_expansion_node import last_graph_data, _save_static_subgraph
            from app.ai.ai_services.graph_expansion import GraphExpansionService
            from app.core.singleton import get_graph_expansion_service
            if early_graph_sent:
                print(
                    "DEBUG WS graph_data: skipped final send (early_graph_sent)",
                    flush=True,
                )
            elif last_graph_data:
                svc = get_graph_expansion_service()
                if svc and relevant_source_documents:
                    exps_for_cited = {}
                    last_exps = getattr(svc, '_last_expansions', None) or {}
                    print(f"DEBUG graph_data: last_exps has {len(last_exps)} entries, relevant_docs={len(relevant_source_documents)}", flush=True)
                    for doc in relevant_source_documents:
                        url = doc.metadata.get("url", "")
                        if url in last_exps:
                            exp = last_exps[url]
                            exps_for_cited[url] = exp
                            print(f"DEBUG graph_data: matched {url[:50]} -> {len(exp.relations)} relations, {len(exp.topics)} topics", flush=True)
                    print(f"DEBUG graph_data: exps_for_cited={len(exps_for_cited)} docs matched", flush=True)
                    cited_subgraph = GraphExpansionService.build_subgraph_data(
                        data.message, relevant_source_documents, exps_for_cited
                    )
                    try:
                        site = state.get("_site", "diplomacy.edu") if 'state' in dir() else "diplomacy.edu"
                        db = svc._client.database_diplo
                        hop2_names = [n["label"] for n in cited_subgraph["nodes"] if n.get("type") not in ("Question", "Source")]
                        print(f"DEBUG hop3: {len(hop2_names)} hop2 entities to expand, db={db}", flush=True)
                        if hop2_names:
                            entity_rels = await svc._client.get_entity_relations(hop2_names, db, limit_per_entity=15)
                            print(f"DEBUG hop3: get_entity_relations returned {len(entity_rels)} entities with rels, total rels={sum(len(v) for v in entity_rels.values())}", flush=True)
                            node_ids = set(n["id"] for n in cited_subgraph["nodes"])
                            edge_set = set((e["from"], e["to"]) for e in cited_subgraph["edges"])
                            colors = GraphExpansionService.SUBGRAPH_COLORS
                            label_prefixes = {"Topic": "t_", "TopicBasket": "t_", "Person": "p_", "Actor": "a_", "Country": "c_", "Process": "t_", "Expert": "p_", "Event": "e_", "Tag": "tag_", "Date": "d_"}
                            all_pfx = list(label_prefixes.values()) + ["n_", "doc_"]
                            def _find(name):
                                for pfx in all_pfx:
                                    nid = f"{pfx}{name}"[:60]
                                    if nid in node_ids:
                                        return nid
                                return None
                            added = 0
                            max_hop3 = 15
                            for ename, rels in entity_rels.items():
                                if added >= max_hop3:
                                    break
                                from_id = _find(ename)
                                if not from_id:
                                    continue
                                for rel in rels:
                                    if added >= max_hop3:
                                        break
                                    tgt_labels = [l for l in (rel.target_labels or []) if l != "Document"]
                                    tgt_type = next((l for l in tgt_labels if l in colors), None)
                                    if not tgt_type and rel.target_post_type:
                                        pt = rel.target_post_type.capitalize()
                                        if pt in colors:
                                            tgt_type = pt
                                    if not tgt_type:
                                        tgt_type = "Topic"
                                    prefix = label_prefixes.get(tgt_type, "n_")
                                    to_id = _find(rel.target_name) or f"{prefix}{rel.target_name}"[:60]
                                    if to_id not in node_ids:
                                        added += 1
                                        cited_subgraph["nodes"].append({
                                            "id": to_id, "label": rel.target_name, "type": tgt_type,
                                            "color": colors.get(tgt_type, "#8d9191"), "size": 5, "hop3": True,
                                            "url": rel.target_url or "",
                                        })
                                        node_ids.add(to_id)
                                    ek = (from_id, to_id)
                                    if ek not in edge_set and from_id != to_id:
                                        edge_set.add(ek)
                                        cited_subgraph["edges"].append({
                                            "from": from_id, "to": to_id,
                                            "label": rel.relationship.replace("_", " ").lower()[:20],
                                            "hop": 3,
                                        })
                            print(f"DEBUG graph_data: hop3 expansion added {added} new nodes", flush=True)
                    except Exception as e3:
                        print(f"WARNING: hop3 expansion failed: {e3}", flush=True)
                    html_path = _save_static_subgraph(
                        data.message, relevant_source_documents, exps_for_cited, svc,
                        citation_urls=citation_urls,
                        graph_override=cited_subgraph,
                    )
                    if html_path:
                        cited_subgraph["subgraph_url"] = html_path
                    await Singleton().websocket_manager.send_message({"status": "graph_data", "text": cited_subgraph}, websocket)
                else:
                    await Singleton().websocket_manager.send_message({"status": "graph_data", "text": last_graph_data}, websocket)
        except Exception as e:
            import traceback
            print(f"WARNING: graph_data send failed: {e}\n{traceback.format_exc()}", flush=True)

        await Singleton().websocket_manager.send_message({"status": "related_questions", "text": related_questions}, websocket)

        # Send renumber_map and citation_urls for clickable citations
        renumber_map_str = {str(k): v for k, v in renumber_map.items()} if renumber_map else {}
        citation_urls_str = {str(k): v for k, v in citation_urls.items()} if citation_urls else {}
        if renumber_map_str:
            await Singleton().websocket_manager.send_message({"status": "renumber_map", "text": renumber_map_str}, websocket)
        if citation_urls_str:
            await Singleton().websocket_manager.send_message({"status": "citation_urls", "text": citation_urls_str}, websocket)
                
        # MongoDB operations - wrapped in try/except to not break the response if DB is down
        timestamp = datetime.now().isoformat()
        message_id = None
        
        try:
            message = self.chat_repository.save(user_ip=data.user_ip, message=data.message, timestamp=timestamp)
            response = self.response_repository.save(message=message, response=answer, response_time=response_time)
            self.__record_conversation(conversation_id, message, response)
            message_id = str(message.id)
        except Exception as e:
            print(f"WARNING: MongoDB save failed: {type(e).__name__}: {str(e)}", flush=True)
            # Generate a temporary message_id for frontend
            import uuid
            message_id = f"temp_{uuid.uuid4().hex[:12]}"
        
        await Singleton().websocket_manager.send_message({"status": "message_id", "text": message_id}, websocket)

        clear_retrieval_overrides()

        return {
            "message_id": message_id,
            "answer": answer,
            "response_time": response_time,
            "sources": sources,
            "related_questions": related_questions,
            "citation_urls": citation_urls,
            "renumber_map": renumber_map
        }
    # called in routes/chat_route.py
    async def handle_chat_request(self, conversation_id: str, data: ChatRouteResponse):
        user_type_str = data.user_type.value if hasattr(data.user_type, 'value') else str(data.user_type)
        original_message = data.message

        # Apply per-request retrieval overrides from frontend
        if hasattr(data, 'retrieval_config') and data.retrieval_config:
            overrides = {k: v for k, v in data.retrieval_config.dict().items() if v is not None}
            set_retrieval_overrides(overrides)
            print(f"DEBUG chat_service (HTTP): Retrieval overrides set: {overrides}", flush=True)
        else:
            overrides = {}
            clear_retrieval_overrides()

        filter_metadata = _filter_langfuse_metadata(overrides)
        config = build_langfuse_run_config(
            thread_id=conversation_id,
            user_id=data.user_ip,
            run_name="chat_http",
            tags=[user_type_str],
            extra_metadata=filter_metadata,
        )
        # this gives the chat id - so we can track the conversation
        thread_config = build_thread_config(conversation_id)
        
        # Set user profile for label-based scoring
        self.label_retriever.set_user_type(data.user_type.value if hasattr(data.user_type, 'value') else str(data.user_type))

        memory = [{'role': c['role'], 'content': c['content']} for c in  self.conversation_history[conversation_id]]
        
        system_prompt = _resolve_system_prompt(data)
        history = repack_chat_history(memory, system_prompt)
        state_update = {'messages': [SystemMessage(content=system_prompt)]}
        await self.diplomacy_bot.aupdate_state(thread_config, state_update)

        start_time = datetime.now()
        
        # Collect answer and docs from stream events (same as WebSocket version)
        answer = ''
        retrieved_docs = []

        stream_input = {'messages': data.message}
        async for event in self.diplomacy_bot.astream_events(stream_input, config=config, version='v2'):
            event_type = event['event']
            if event_type == 'on_chat_model_stream':
                content = event['data']['chunk'].content
                if content:
                    if isinstance(content, list) and len(content) > 0 and 'text' in content[0].keys():
                        text = content[0]['text']
                    else:
                        text = content
                    answer += text
            elif event_type == 'on_retriever_end':
                # Only capture first retriever result
                if not retrieved_docs:
                    reranked = getattr(self.label_retriever, '_last_reranked_docs', None)
                    if reranked:
                        retrieved_docs = reranked
                        self.label_retriever._last_reranked_docs = None
                    else:
                        retrieved_docs = event['data']['output']
        
        _lf_handler = get_langfuse_callback_handler()
        patch_langfuse_trace_metadata(
            filter_metadata,
            trace_id=getattr(_lf_handler, "last_trace_id", None) if _lf_handler else None,
        )
        
        # Filter responses
        relevant_source_documents, related_questions, citation_urls, renumber_map = self.filter_related_sources(
            retrieved_docs, data.message, answer,
            langfuse_session_id=conversation_id,
            langfuse_user_id=data.user_ip,
        )
        
        end_time = datetime.now()

        response_time = (end_time - start_time).total_seconds()

        sources = self.__repack_source_documents(relevant_source_documents, citation_urls)
                
        timestamp = datetime.now().isoformat()
        
        try:
            message = self.chat_repository.save(user_ip=data.user_ip, message=data.message, timestamp=timestamp)
            response = self.response_repository.save(message=message, response=answer, response_time=response_time)
            self.__record_conversation(conversation_id, message, response)
            message_id = str(message.id)
        except Exception as e:
            print(f"WARNING: MongoDB save failed: {type(e).__name__}: {str(e)}", flush=True)
            import uuid
            message_id = f"temp_{uuid.uuid4().hex[:12]}"

        clear_retrieval_overrides()

        graph_data_payload = None
        try:
            from app.ai.ai_services.graph_expansion_node import last_graph_data
            if last_graph_data:
                graph_data_payload = {
                    "nodes": last_graph_data.get("nodes"),
                    "edges": last_graph_data.get("edges"),
                    "subgraph_url": last_graph_data.get("subgraph_url"),
                }
        except Exception as e:
            print(f"WARNING: graph_data attach (HTTP) failed: {type(e).__name__}: {e}", flush=True)

        return {
            "message_id": message_id,
            "answer": answer,
            "response_time": response_time,
            "sources": sources,
            "related_questions": related_questions,
            "citation_urls": citation_urls,
            "renumber_map": renumber_map,
            "graph_data": graph_data_payload,
        }
    
       
       
    def filter_related_sources(
        self,
        source_documents,
        question,
        answer,
        live_renumber_map=None,
        deep_link_service=None,
        *,
        langfuse_session_id: str | None = None,
        langfuse_user_id: str | None = None,
    ):
        """
        Filter and return source documents based on citations in the answer.
        
        Args:
            source_documents: List of retrieved documents
            question: User's question
            answer: LLM's answer (already renumbered if live_renumber_map provided)
            live_renumber_map: Optional dict {orig_num: new_num} from live streaming renumbering
                              If provided, answer already has sequential citations [1], [2], [3]...
            deep_link_service: Optional DeepLinkService instance for Redis shortening (reused across docs)
        """
        logger.debug(f"filter_related_sources called with {len(source_documents) if source_documents else 0} documents")
        
        # Helper function to deduplicate documents by URL for related questions extraction
        def dedupe_by_url(docs):
            seen_urls = set()
            unique_docs = []
            for doc in docs:
                url = get_metadata_field(doc.metadata, 'url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_docs.append(doc)
                elif not url:
                    unique_docs.append(doc)
            return unique_docs
        
        # Helper to extract related questions from documents
        def safe_get_questions(doc):
            try:
                node_content = doc.metadata.get('_node_content', '')
                if not node_content:
                    return ''
                parsed = json.loads(node_content)
                return parsed.get('metadata', {}).get('questions_this_excerpt_can_answer', '')
            except (json.JSONDecodeError, AttributeError, TypeError):
                return ''
        
        def extract_related_questions(docs):
            if len(docs) == 1:
                return [q.strip('123. ') for r in docs for q in safe_get_questions(r).split('\n')[:4] if '?' in q]
            elif len(docs) == 2:
                return [q.strip('123. ') for r in docs for q in safe_get_questions(r).split('\n')[:2] if '?' in q]
            elif len(docs) > 2:
                return [q.strip('123. ') for r in docs for q in safe_get_questions(r).split('\n')[:1] if '?' in q]
            return []
        
        # =============================================================================
        # CITE_SOURCES: Use live_renumber_map if provided (streaming already renumbered)
        # Otherwise parse citations from answer
        # NO DEDUPLICATION - show all chunks as separate sources
        # =============================================================================
        _cite = get_param('cite_sources', CITE_SOURCES)
        if _cite:
            citation_urls = {}
            renumber_map = {}  # Will be empty if live renumbering was done
            
            # Determine which original documents were cited
            if live_renumber_map:
                # Live renumbering was done - answer already has [1], [2], [3]...
                # live_renumber_map = {orig_num: new_num}, e.g. {1: 1, 4: 2, 6: 3}
                # We need to get original indices from the keys
                cited_orig_nums = sorted(live_renumber_map.keys(), key=lambda k: live_renumber_map[k])
                cited_indices = [n - 1 for n in cited_orig_nums]  # Convert to 0-based, ordered by appearance in answer
                logger.debug(f"CITE_SOURCES (live): Using live_renumber_map {live_renumber_map}, cited_indices = {cited_indices}")
            else:
                # No live renumbering - parse citations from answer
                cited_indices = parse_citations(answer)
                logger.debug(f"CITE_SOURCES: Found citations {cited_indices} in answer")
            
            _dl_svc = deep_link_service or (DeepLinkService() if get_param('use_short_deep_links', USE_SHORT_DEEP_LINKS) else None)
            
            if cited_indices:
                # Collect cited documents in the order they were cited
                cited_docs = []
                for i, old_idx in enumerate(cited_indices):
                    if 0 <= old_idx < len(source_documents):
                        cited_docs.append(source_documents[old_idx])
                        if not live_renumber_map:
                            renumber_map[old_idx + 1] = i + 1
                
                # Batch deep link creation for all cited docs
                batch_urls = build_deep_link_urls_batch(cited_docs, query=question, deep_link_service=_dl_svc)
                citation_urls = {num: batch_urls[num] for num in batch_urls}
                
                logger.debug(f"CITE_SOURCES: Returning {len(cited_docs)} sources (no deduplication)")
                logger.debug(f"CITE_SOURCES: renumber_map = {renumber_map}")
                logger.debug(f"CITE_SOURCES: citation_urls = {citation_urls}")
                
                related_questions = list(dict.fromkeys(extract_related_questions(cited_docs)))
                return cited_docs, related_questions, citation_urls, renumber_map
            else:
                # No citations found - return all sources as fallback
                logger.debug("CITE_SOURCES: No citations found, returning all sources")
                citation_urls = build_deep_link_urls_batch(source_documents, query=question, deep_link_service=_dl_svc)
                related_questions = list(dict.fromkeys(extract_related_questions(source_documents)))
                return source_documents, related_questions, citation_urls, {}
        
        # =============================================================================
        # USE_SOURCE_FILTER=False: Return all sources without filtering
        # NO DEDUPLICATION - show all chunks as separate sources
        # =============================================================================
        if not get_param('use_source_filter', USE_SOURCE_FILTER):
            related_questions = list(dict.fromkeys(extract_related_questions(source_documents)))
            # No citation_urls when CITE_SOURCES is disabled
            return source_documents, related_questions, {}, {}
        
        # =============================================================================
        # USE_SOURCE_FILTER=True: Use LLM to filter sources (legacy, adds latency)
        # =============================================================================
        resources = '\n-------\n'.join([r.page_content for r in source_documents])
        reosurce_instruction = 'This is the list of texts retrieved from database. Your task is to return the list of booleans, where True would be if the text is used for generating the answer, and False otherwise. The form of the answer should be: [True, False, True, ...]. Do not explain anything, and do not return anything except the list.'
        response = self.resource_filter.invoke(
            [
                {'role': 'human', 'content': f"Question: {question}\nAnswer: {answer}\n\nRetrieved texts:\n{resources}\n\n"},
                {'role': 'human', 'content': reosurce_instruction},
            ],
            config=langfuse_invoke_config(
                thread_id=langfuse_session_id,
                user_id=langfuse_user_id,
                run_name="source_filter",
            ),
        )
        filter_result = ast.literal_eval(response.content)
        relevant_source_documents = [sc for sc, f in zip(source_documents, filter_result) if f]
        unique_docs = dedupe_by_url(relevant_source_documents)
        related_questions = list(dict.fromkeys(extract_related_questions(unique_docs)))
        
        # No citation_urls when using LLM filter
        return relevant_source_documents, related_questions, {}, {}
     
    def __repack_source_documents(self, relevant_source_documents, citation_urls=None):
        """
        Repack source documents into ResponseSource objects for the frontend.
        
        Args:
            relevant_source_documents: List of Document objects
            citation_urls: Optional dict {1-based_index: deep_link_url} with pre-built deep link URLs
        """
        if citation_urls is None:
            citation_urls = {}
        
        packed = [{'text': d.page_content, **d.metadata} for d in relevant_source_documents]

        from app.core.weaviate_props import present_source_fields

        # Recover title/url from oneweaviate encoded filenames before deep-link build.
        for i, s in enumerate(packed):
            display_title, recovered_url = present_source_fields(s)
            if display_title:
                s["title"] = display_title
                if not s.get("name"):
                    s["name"] = display_title
            href = recovered_url or s.get("url") or s.get("link") or ""
            if href:
                s["url"] = href
                if not s.get("link"):
                    s["link"] = href

            if s.get("link", "").startswith("http://"):
                s["link"] = "https://" + s["link"][7:]

            raw_url = s.get("url", "")
            if raw_url:
                normalized_url, _frag = _normalize_url_with_redirect(raw_url)
                s["url"] = normalized_url + _frag

            # Build the href before s["text"] becomes the deep-link payload,
            # otherwise the payload gets wrapped into itself a second time.
            s["_card_href"] = _source_card_deep_link(
                s, preferred=citation_urls.get(i + 1)
            )
            s["text"] = _build_deep_link_text(s["text"], s, s.get("title", ""), s.get("url", ""))

        return [
            ResponseSource(
                text=s["text"],
                title=s.get("title") or s.get("name") or "Untitled",
                date=str(s.get("date")) if s.get("date") else "Unknown date",
                url=s.get("url") or s.get("link") or "",
                link=s.get("link"),
                name=s.get("name"),
                deep_link_url=s.get("_card_href"),
            )
            for s in packed
        ]
        
    def __record_conversation(self, conversation_id: str,  message: ChatMessageModel, response: ChatResponseModel):
        
        self.conversation_history[conversation_id].extend([
            {'role': 'user', 'content': f"```{message.message}```", 'timestamp': message.timestamp},
            {'role': 'assistant', 'content': response.response, 'timestamp': message.timestamp}
        ])
        
    def validate_message(self, message_id: str):
        if self.chat_repository.get_by_id(message_id) is None:
            raise ValueError("Message not found")
        
    def validate_response(self, message_id: str):
        if self.response_repository.get_by_message_id(message_id) is None:
            raise ValueError("Response not found")
    
    def validate_feedback_type(self, feedback_type: int):
        if feedback_type not in [-1, 0, 1]:
            raise ValueError("Invalid feedback type")
        
    def save_feedback(self, message_id: str, feedback: FeedbackRouteRequest):
        response = self.response_repository.get_by_message_id(message_id)
        
        response.feedback_type = feedback.feedback_type
        response.feedback_message = feedback.feedback_message
        response.save()


def get_chat_service() -> ChatService:
    return ChatService(use_sentence_header=False)


def get_chat_service_sentence_header() -> ChatService:
    return ChatService(use_sentence_header=True)
