"""
LangGraph agent definition for the Diplomacy chatbot.

Builds the conversational RAG pipeline:
- Retriever selection via factory
- Reranker integration
- Post-reranker label weight application
- Graph compilation (direct retrieval or full ReAct)
"""

import os
import time
import uuid
from typing import Annotated, List
# urllib module - for the URL parsing - this makes it easier to work with URLs
from urllib.parse import urlparse

import weaviate
from typing_extensions import TypedDict
# langchain modules
# documents module - for the documents that are retrieved from the database
from langchain_core.documents import Document 
# embeddings module - for the embeddings of the documents
from langchain_core.embeddings import Embeddings
# messages module - for the messages that are sent to the LLM
from langchain_core.messages import (
    SystemMessage, ToolMessage, AIMessage, AIMessageChunk, HumanMessage,
    filter_messages, trim_messages,
)
# custom events (rag_progress) → chat_service maps them to WS status=progress
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
# prompts module - for the prompts that are used to generate the response
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# openai module - for the OpenAI LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# anthropic module - for the Anthropic LLM
from langchain_anthropic import ChatAnthropic
# tools module - for the tools that are used to generate the response
from langchain.tools import tool
# langgraph modules
# graph module - for the graph that is used to generate the response
from langgraph.graph import StateGraph, START, END
# message module - for the messages that are sent to the LLM
from langgraph.graph.message import add_messages
# prebuilt module - for the prebuilt tools that are used to generate the response
from langgraph.prebuilt import ToolNode, tools_condition
# checkpoint module - for the checkpoint that is used to save the state of the graph
from langgraph.checkpoint.memory import InMemorySaver
# .env config file
from app.core.config import (
    INDEX_NAME, INDEX_NAME2,
    LLM_MODEL_NAME, LLM_MODEL_PROVIDER, EMBEDDING_MODEL_NAME,
    LOCAL_LLM_URL, LOCAL_LLM_MODEL, LOCAL_LLM_KEY,
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
    OPENAI_KEY, OPENAI_ORGANIZATION, WV_KEY, WV_CLIENT_URL, WV_GRPC_PORT,
    MAX_TOOL_CALLS, SKIP_TOOL_DECISION, RETRIEVAL_CHUNKS, CITE_SOURCES,
    EMBEDDING_PROVIDER, LOCAL_EMBEDDING_URL, LOCAL_EMBEDDING_KEY,
    USE_RERANKER, RERANKER_URL, RERANKER_API_KEY, RERANKER_TOP_K,
    RETRIEVAL_CACHE_TTL,
    USE_DYNAMIC_LABEL_WEIGHTS, DYNAMIC_WEIGHT_ALPHA, DYNAMIC_WEIGHT_MIN_SIM,
    LLM_FAILSAFE_ENABLED, FAILSAFE_LLM_PROVIDER, FAILSAFE_LLM_MODEL,
    FAILSAFE_LLM_URL, FAILSAFE_LLM_KEY,
)
# custom modules
from app.ai.ai_services.embeddings import TEIEmbeddings
from app.ai.ai_services.reranker import TEIReranker
from app.ai.ai_services.label_weights import DEFAULT_LABEL_WEIGHT, get_label_weights
from app.ai.ai_services.query_intent import initialize_prototypes, get_dynamic_label_weights
from app.ai.ai_services.retrieval_cache import _cache_key, _serialize_docs, _deserialize_docs
from app.ai.ai_services.retrievers.factory import create_retriever, resolve_retrieval_mode
from app.ai.ai_services.retrievers.recency import get_recency_multiplier_for_metadata
from app.core.retrieval_context import get_param
from app.debugg.llm_model_used import log_model_configured

# orchestrates pipeline for the chatbot

# ---------------------------------------------------------------------------
# Weaviate client (module-level singleton)
# ---------------------------------------------------------------------------
# description: _parsed_url parses the URL of the weaviate database
_parsed_url = urlparse(WV_CLIENT_URL)
# hostname + HTTP port must be explicit in WV_CLIENT_URL (e.g. http://host:8591)
_weaviate_host = _parsed_url.hostname
if not _weaviate_host:
    raise ValueError("WV_CLIENT_URL must include a hostname")
if _parsed_url.port is None:
    raise ValueError(
        "WV_CLIENT_URL must include a port, e.g. http://ai6000.diplomacy.edu:8591"
    )
_weaviate_port = _parsed_url.port

# gRPC port must come from .env (WV_GRPC_PORT) — no hardcoded fallbacks per env
if WV_GRPC_PORT <= 0:
    raise ValueError(
        "WV_GRPC_PORT must be set in .env (e.g. 50153 for chatbot-api-dev, "
        "50152 for chatbot-api prod)"
    )
_weaviate_grpc_port = WV_GRPC_PORT
# debug message - prints the URL of the weaviate database
print(f"INFO: Connecting to Weaviate at {_weaviate_host}:{_weaviate_port} "
      f"(gRPC: {_weaviate_grpc_port})", flush=True)
# connects to the weaviate database - this is a weaviate module that is used to connect to the weaviate database
weaviate_client = weaviate.connect_to_local(
    host=_weaviate_host,
    port=_weaviate_port,
    grpc_port=_weaviate_grpc_port,
    auth_credentials=weaviate.auth.AuthApiKey(WV_KEY),
    additional_config=weaviate.classes.init.AdditionalConfig(
        timeout=weaviate.classes.init.Timeout(init=30)
    ),
)

# returns the weaviate client
def get_weaviate_client():
    return weaviate_client


# ---------------------------------------------------------------------------
# Global reranker instance
# ---------------------------------------------------------------------------
# description: initializes the TEI reranker if it is enabled in the config
# takes params:
    # url: the URL of the TEI reranker 
    # api_key: the API key for the TEI reranker inside .env file
    # top_k: the number of documents to rerank 
# returns:
    # the TEI reranker instance - var that holds TEI reranker functionalityes that can be called in other functions
tei_reranker = None
if USE_RERANKER and RERANKER_URL and RERANKER_API_KEY:
    tei_reranker = TEIReranker(
        url=RERANKER_URL,
        api_key=RERANKER_API_KEY,
        top_k=RERANKER_TOP_K,
    )
    print(f"INFO: TEI Reranker initialized with URL: {RERANKER_URL}", flush=True)


# ---------------------------------------------------------------------------
# LLM factory with failsafe fallback
# ---------------------------------------------------------------------------

# description: _create_llm creates the LLM instance - the LLM model that will be used to generate the response
# takes params:
    # provider: the provider of the LLM (local, openai, anthropic, deepseek)
    # model_name: the name of the LLM model (default is None)
    # temperature: the temperature of the LLM (default is None)
# returns:
    # the LLM instance
def _create_llm(provider, model_name=None, temperature=None):
    """Create a single ChatModel instance for a given provider."""
    # extra is a dictionary that is used to pass additional arguments to the LLM instance
    # this is set in WordPress frontend - and passed to the chatbot
    extra = {}
    # temperature is for creativity of the LLM
    if temperature is not None:
        extra['temperature'] = temperature
    # this is for logging the model that is used
    log_model_configured(
        provider,
        model_name or (LOCAL_LLM_MODEL if provider == "local" else LLM_MODEL_NAME),
    )
    # if provider is anthropic, it creates the Anthropic LLM instance
    if provider == "anthropic":
        return ChatAnthropic(model=model_name or LLM_MODEL_NAME, **extra)
    # if provider is openai, it creates the OpenAI LLM instance
    elif provider == "openai":
        return ChatOpenAI(
            model_name=model_name or LLM_MODEL_NAME,
            api_key=OPENAI_KEY, organization=OPENAI_ORGANIZATION, **extra)
    # if provider is deepseek, it creates the DeepSeek LLM instance (OpenAI-compatible API)
    elif provider == "deepseek":
        return ChatOpenAI(
            model_name=model_name or LLM_MODEL_NAME,
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL, **extra)
    # if provider is local, it creates the local LLM instance
    elif provider == "local":
        return ChatOpenAI(
            model_name=model_name or LOCAL_LLM_MODEL,
            api_key=LOCAL_LLM_KEY, base_url=LOCAL_LLM_URL, **extra)
    else:
        raise ValueError(f"Invalid LLM provider: {provider}")

# description: _create_failsafe_llm creates the fallback LLM in case the main LLM fails (helper function called in _with_failsafe)
# takes params:
    # temperature: the temperature of the fallback LLM (default is None)
# returns:
    # the fallback LLM instance
def _create_failsafe_llm(temperature=None):
    """Create the failsafe fallback LLM using FAILSAFE_LLM_* config."""
    extra = {}
    if temperature is not None:
        extra['temperature'] = temperature
    # this is config for the fallback LLM that is used in case the main LLM fails
    fs_provider = FAILSAFE_LLM_PROVIDER
    fs_model = FAILSAFE_LLM_MODEL or LLM_MODEL_NAME
    # local is the LLM that is used as fallback
    if fs_provider == "local":
        return ChatOpenAI(
            model_name=FAILSAFE_LLM_MODEL or LOCAL_LLM_MODEL,
            api_key=FAILSAFE_LLM_KEY or LOCAL_LLM_KEY,
            base_url=FAILSAFE_LLM_URL or LOCAL_LLM_URL, **extra)
    elif fs_provider == "anthropic":
        return ChatAnthropic(model=fs_model, **extra)
    elif fs_provider == "deepseek":
        return ChatOpenAI(
            model_name=fs_model, api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_URL, **extra)
    else:
        return ChatOpenAI(
            model_name=fs_model, api_key=OPENAI_KEY,
            organization=OPENAI_ORGANIZATION, **extra)

# description: _failsafe_active checks if the failsafe is activated (different provider or model configured)
# takes params:
    # None (no params) because it is a helper function
# returns:
    # True if the failsafe is activated
    # False if the failsafe is not activated
def _failsafe_active():
    """Check if failsafe should be applied (different provider or model configured)."""
    if not (LLM_FAILSAFE_ENABLED and FAILSAFE_LLM_PROVIDER and
            (FAILSAFE_LLM_PROVIDER != LLM_MODEL_PROVIDER or FAILSAFE_LLM_MODEL)):
        return False
    # Skip failsafe when required credentials are missing (e.g. local-only .env).
    if FAILSAFE_LLM_PROVIDER == "openai" and not OPENAI_KEY:
        return False
    if FAILSAFE_LLM_PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
        return False
    if FAILSAFE_LLM_PROVIDER == "local" and not (FAILSAFE_LLM_KEY or LOCAL_LLM_KEY):
        return False
    return True


_failsafe_logged = False

# description: combines the primary LLM with the fallback LLM if the failsafe is activated (helper function called in _create_llm)
# takes params:
    # primary: the primary LLM instance
    # temperature: the temperature of the fallback LLM (default is None)
# returns:
    # the primary LLM instance with the fallback LLM if the failsafe is activated
def _with_failsafe(primary, temperature=None):
    """Wrap a Runnable with a failsafe fallback if enabled. Works with any Runnable."""
    global _failsafe_logged
    # check if the failsafe is activated
    if _failsafe_active():
        fallback = _create_failsafe_llm(temperature=temperature)
        if not _failsafe_logged:
            print(f"INFO: LLM failsafe enabled: {LLM_MODEL_PROVIDER} -> "
                  f"{FAILSAFE_LLM_PROVIDER} ({FAILSAFE_LLM_MODEL or LLM_MODEL_NAME})",
                  flush=True)
            _failsafe_logged = True
        # if faillsafe is activated, return the primary LLM instance with the fallback LLM
        return primary.with_fallbacks([fallback])
    # else return the primary LLM instance if the failsafe is not activated
    return primary


# ---------------------------------------------------------------------------
# Shared retrieval helpers
# ---------------------------------------------------------------------------
# description: apply_post_reranker_label_weights applies the post-reranker label weights to the documents - its now helper function for both graphs
# parameters:
# - docs: the documents that are retrieved from the database - inside of langchain Document class
# - top_k: the number of documents to return - input parameter from the graph.py
# - active_retriever: the active retriever that is used to retrieve the documents from the database
# - query_vector: the query vector that is used to retrieve the documents from the database
# returns:
# - the documents with the post-reranker label weights applied (list of documents)
def apply_post_reranker_label_weights(
    docs: List[Document],
    top_k: int,
    active_retriever,
    query_vector: list[float] | None = None,
) -> List[Document]:
    """Re-score docs by label + recency weights; return top_k."""
    # user_type is the type of the user - general is the default type
    user_type = "general"
    if hasattr(active_retriever, 'user_type'):
        user_type = active_retriever.user_type
    # _use_dynamic is a boolean that is used to check if the dynamic label weights are enabled - input parameter from the graph.py
    _use_dynamic = get_param('use_dynamic_label_weights', USE_DYNAMIC_LABEL_WEIGHTS)
    #
    intent_sims = {}
    # if the dynamic label weights are enabled and the query vector is provided, then get the dynamic label weights
    if _use_dynamic and query_vector:
        _alpha = get_param('dynamic_weight_alpha', DYNAMIC_WEIGHT_ALPHA)
        _min_sim = get_param('dynamic_weight_min_sim', DYNAMIC_WEIGHT_MIN_SIM)
        label_weights, intent_sims = get_dynamic_label_weights(
            query_vector, user_type, alpha=_alpha, min_similarity=_min_sim
        )
    # if the intent similarities are not empty, then print the top similarities
        if intent_sims:
            top_intents = sorted(intent_sims.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"INTENT: Dynamic weights active. Top similarities: "
                  f"{', '.join(f'{pt}={s:.3f}' for pt, s in top_intents)}", flush=True)
    # if the dynamic label weights are not enabled, then get the default label weights from the user type
    else:
        label_weights = get_label_weights(user_type)
    # loop through passed documents and apply the post-reranker label weights
    for doc in docs:
        label = doc.metadata.get('label', '')
        lw = label_weights.get(label, DEFAULT_LABEL_WEIGHT)
    # get the base score from the document metadata - this is the score from the reranker
        base_score = doc.metadata.get('_reranker_score',
                     doc.metadata.get('_final_score',
                     doc.metadata.get('_rescored', 0)))
        # get the recency multiplier from the document metadata - this is the recency multiplier for the document
        rm = get_recency_multiplier_for_metadata(doc.metadata)
        # calculate the post-reranker score
        post = base_score * lw * rm
        # set the label weight, recency multiplier, base score, and post-reranker score in the document metadata
        doc.metadata['_label_weight'] = lw
        doc.metadata['_recency_multiplier'] = rm
        doc.metadata['_pre_label_score'] = base_score
        doc.metadata['_post_label_score'] = post
    # sort the documents by the post-reranker score in descending order
    docs.sort(key=lambda d: d.metadata.get('_post_label_score', 0), reverse=True)
    # debugg message witl all metadata for the documents
    print(f"DEBUG: Post-reranker label weights applied (profile={user_type}). All {len(docs)} docs:", flush=True)
    for i, d in enumerate(docs):
        url = d.metadata.get('url') or ''
        label = d.metadata.get('label') or '?'
        pre = d.metadata.get('_pre_label_score', 0)
        post = d.metadata.get('_post_label_score', 0)
        lw = d.metadata.get('_label_weight', 1.0)
        rm = d.metadata.get('_recency_multiplier', 1.0)
        marker = ' <<<' if i < top_k else ''
        title = d.metadata.get('title') or ''
        print(f"DEBUG:   {i+1:2d}. [{post:.4f}] (reranker={pre:.4f} x lw={lw} x rm={rm:.3f}) "
              f"{label:10s} {title[:40]} | {url[:60]}{marker}", flush=True)
        if i < top_k and '/topics/' in url:
            content = d.page_content.replace('\n', ' | ')
            print(f"DEBUG:   ^^^ TOPIC CONTENT ({len(d.page_content)} chars): {content[:800]}", flush=True)

    return docs[:top_k]
# context: direct_retrieval_node is called by the graph.py file - this is the function that gives data to the direct retrieval node
# description: make_direct_retrieval_node creates the direct retrieval node - which is the main node for the graph (shared retrieve once)
# parameters:
    # active_retriever: it is called from create_retriever function in graph.py
    # mode_reranker_top_k: it is called from create_retriever function in graph.py
# returns:
    # the direct retrieval node
async def _astream_llm_response(llm, messages) -> AIMessage:
    """Stream LLM tokens (surface via astream_events for WS); return full AIMessage for state."""
    response = None
    async for chunk in llm.astream(messages):
        response = chunk if response is None else response + chunk
    if response is None:
        return AIMessage(content="")
    if isinstance(response, AIMessageChunk):
        return AIMessage(
            content=response.content,
            additional_kwargs=response.additional_kwargs or {},
            response_metadata=response.response_metadata or {},
            id=response.id,
        )
    return response if isinstance(response, AIMessage) else AIMessage(content=str(response))

# context: called by adispatch_custom_event in graph.py 
# description: _doc_progress_items is a helper function that creates the source cards for the rag_progress sources_ready event
# parameters:
    # docs: the documents that are retrieved from the database - inside of langchain Document class
    # limit: the number of documents to return - default is 8
# returns:
    # a list of items - each item is a source card
def _doc_progress_items(docs: list, limit: int = 8) -> list[dict[str, str]]:
    """Lightweight source cards for the rag_progress sources_ready event."""
    # make a list of items -
    items: list[dict[str, str]] = []
    for d in docs[:limit]:
        # get the metadata from the document - this is the metadata of the document
        meta = getattr(d, "metadata", None) or {}
        # get the page content from the document - this is the content of the document
        text = getattr(d, "page_content", "") or ""
        # append the item to the list of items
        from app.core.weaviate_props import present_source_fields

        display_title, recovered_url = present_source_fields(meta)
        href = recovered_url or meta.get("url") or meta.get("link") or ""
        items.append(
            {
                "title": (display_title or meta.get("title") or meta.get("name") or "Untitled")[:120],
                "url": href,
                "text": text[:800],
                "date": str(meta.get("date") or "")[:40],
            }
        )
    return items


def make_direct_retrieval_node(active_retriever, mode_reranker_top_k: int):
    """Factory for the direct_retrieval LangGraph node (shared retrieve once)."""
# description: direct_retrieval_node is the main node for the graph (shared retrieve once)
    async def direct_retrieval_node(state):
# importing logger - this needs to go up
        from app.core.logging import logger
        # get messages from the class State - this is from user + system prompt - at this point we got the user question and system prompt
        messages = state['messages']
        # make var for user question - that way we can extract it from state ['messages']
        user_question = None
        # loop through messages in reverse order - last message is the user question - and get the user question 
        # - if found break the loop - user_question = msg.content
        # - if not found, get the last message as user question (messages[-1])
        for msg in reversed(messages):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                user_question = msg.content
                break
        if not user_question:
            user_question = str(messages[-1]) if messages else ""
        # logg the user question - but limit to 100 characters - that way DEBUG is not too long
        logger.debug(f"direct_retrieval_node: Retrieving for question: {user_question[:100]}")
        # get the cache ttl from the config - this brings back the number of seconds to cache the documents - set in config.py - if more than 0 ⬇️
        # ================== params for retrieval ==================
        _cache_ttl = get_param('retrieval_cache_ttl', RETRIEVAL_CACHE_TTL)
        # get the retrieval chunks from the config - this brings back the number of chunks to retrieve - set in config.py
        _chunks = get_param('retrieval_chunks', RETRIEVAL_CHUNKS)
        # get the cite sources from the config - this brings back the boolean if we should cite the sources - set in config.py
        _cite = get_param('cite_sources', CITE_SOURCES)
        # get the use reranker from the config - this brings back the boolean if we should use the reranker - set in config.py 
        _use_reranker = get_param('use_reranker', USE_RERANKER)
        # get the reranker top k from the config - this brings back the number of documents to rerank - set in config.py 
        _reranker_top_k = get_param('reranker_top_k', mode_reranker_top_k)
        # ========================================================
        cache_hit = False
        t_retrieval = 0.0
        t_reranker = 0.0
        docs = []
        # if cache ttl is more than 0, then try to get the documents from Redis
        if _cache_ttl > 0:
            try:
            # import RedisClient from utils.redis_client.py - this is the class that handles the Redis connection
                from app.utils.redis_client import RedisClient
                _redis = RedisClient.get_client()
            # look up the cache key for the user question
                _ckey = _cache_key(user_question)
            # get catched documents from Redis
                _cached = _redis.get(_ckey)
                if _cached:
                    docs = _deserialize_docs(_cached)
                    cache_hit = True
            # log the cache hit
                    print(f"CACHE HIT for '{user_question[:60]}' ({len(docs)} docs)", flush=True)
            except Exception as e:
                print(f"WARNING: Retrieval cache read failed: {e}", flush=True)
        # if there is no cache hit, then retrieve the documents from the database - active_retriever.invoke(user_question)
        if not cache_hit:
            t_ret = time.time()
            docs = active_retriever.invoke(user_question)
            t_retrieval = time.time() - t_ret
            logger.debug(f"direct_retrieval_node: Retrieved {len(docs)} documents before reranking")

            t_rr = time.time()
            if tei_reranker is not None and _use_reranker:
                docs = tei_reranker.rerank(user_question, docs, top_k=_reranker_top_k)
                logger.debug(f"direct_retrieval_node: After reranking: {len(docs)} documents")
            t_reranker = time.time() - t_rr

            # if cache ttl is more than 0, then store the documents in Redis
            if _cache_ttl > 0:
                try:
                    from app.utils.redis_client import RedisClient
                    _redis = RedisClient.get_client()
                    # call the _cache_key function and give it user_questiong from state ['messages']
                    _ckey = _cache_key(user_question)
                    _redis.setex(_ckey, _cache_ttl, _serialize_docs(docs))
                    print(f"CACHE STORE for '{user_question[:60]}' ({len(docs)} docs, "
                          f"TTL={_cache_ttl}s)", flush=True)
                except Exception as e:
                    print(f"WARNING: Retrieval cache write failed: {e}", flush=True)
        
        # from create_retriever function in graph.py - active_retriever is the instance of the chosen retreival mode
        # in that instance we also have last_query_vector - this is the vectorized user question used to look in Weaviate
        # last_query_vector means give me that vector or none if not set
        qv = getattr(active_retriever, 'last_query_vector', None)
        # now on that vector and chunks retreived from Weaviate - apply the post-reranker label weights - give some post_types more importance
        docs = apply_post_reranker_label_weights(
            docs, _chunks, active_retriever, query_vector=qv
        )
        # log the timing of the retrieval and reranking - and if cache hit or miss
        print(f"TIMING direct_retrieval: retrieval={t_retrieval:.3f}s "
              f"reranker={t_reranker:.3f}s cache={'HIT' if cache_hit else 'MISS'}", flush=True)

        # set the last reranked documents in the active retriever instance - this is used to avoid reranking the same documents again
        active_retriever._last_reranked_docs = docs

        # create a unique tool call id - this is used to identify the tool call in the LLM response - this is used to call the tool - diplo_tool
        # 
        tool_call_id = str(uuid.uuid4())
        # create the AI message - this is the message that will be sent to the LLM
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": tool_call_id,
                "name": "diplo_tool",
                "args": {"question": user_question}
            }]
        )
        # if cite sources is enabled, then format the documents to be sent to the LLM
        if _cite:
            formatted_docs = []
            for i, doc in enumerate(docs):
                # from doc.metadata get the title - if not found, use 'Unknown'
                title = doc.metadata.get('title', 'Unknown')
                # format the document to be sent to the LLM - this is the title and the content of the document
                # this is the format: [{number}] {title}\n{content}
                formatted_docs.append(f"[{i+1}] {title}\n{doc.page_content}")
            # tool_content is var that holds number page and content - separated by --- this will be sent as _cite
            tool_content = "\n\n---\n\n".join(formatted_docs)
        else:
            tool_content = str(docs)
        # create the tool_msg - this is the cite message that will be sent to the LLM
        tool_msg = ToolMessage(
            content=tool_content,
            tool_call_id=tool_call_id,
            name="diplo_tool"
        )
        # import WEBSITE_NAME from config.py - this is the name of the website - default is diplomacy.edu
        from app.core.config import WEBSITE_NAME
    # this sums up ai_msg, tool_msg, tool_call_count, _retrieved_docs, _site, _question and returns it to graph.py
        return {
            'messages': [ai_msg, tool_msg],
            'tool_call_count': 1,
            '_retrieved_docs': docs,
            '_site': get_param('site_filter_name', None) or WEBSITE_NAME,
            '_question': user_question,
        }
    # return the direct_retrieval_node function to graph.py
    return direct_retrieval_node


# ---------------------------------------------------------------------------
# Main initialization function
# ---------------------------------------------------------------------------

# context: main function called by the app.ai.ai_services.graph.py - serves as orchestrator for langgraph graph
# - called by g2 = StateGraph(State) in graph.py
# description: initialize_diplomacy_bot_graph initializes the diplomacy bot graph 
    # gets all the components and configs for the graph
# takes params:
    # retrieval_mode: optional override (e.g. 'sentence_header' for A/B /api/chat-sh).
# returns:
    # the compiled graph and the active retriever
def initialize_diplomacy_bot_graph(retrieval_mode: str | None = None):
    """Build and return (compiled_graph, active_retriever).

    retrieval_mode: optional override (e.g. 'sentence_header' for A/B /api/chat-sh).
    When None, uses resolve_retrieval_mode() from config (prod /api/chat path).
    """
    
# description: State is a typed dictionary that defines the state of the graph
    # it shows the configured variables for RAG pipeline
# takes params:
    # None (no params) because it is a helper function
# returns:
    # the State class which are the configured variables for RAG pipeline
    class State(TypedDict):
        messages: Annotated[list, add_messages] # messages in the chat (list of messages) - 
        tool_call_count: int # number of tool calls (int) if there are 
        _retrieved_docs: list # retrieved documents (list)
        _site: str # site of the chat (str)
        _question: str # question asked by the user (str)
        _graph_data: dict # data of the graph (dict)

    # Embeddings - used to embed the documents so it can be used to search vector database
    # it can be local or openai
    if EMBEDDING_PROVIDER == 'local':
        from app.core.logging import logger
        logger.info(f"Using LOCAL embeddings: {LOCAL_EMBEDDING_URL}")
        embeddings = TEIEmbeddings(url=LOCAL_EMBEDDING_URL, api_key=LOCAL_EMBEDDING_KEY)
    else:
        from app.core.logging import logger
        logger.info(f"Using OpenAI embeddings: {EMBEDDING_MODEL_NAME}")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=OPENAI_KEY)

    # Initialize prototype embeddings for dynamic label weights
    # labeel weights are used to give cetrtain post type more importance in the retrieval results
    if get_param('use_dynamic_label_weights', USE_DYNAMIC_LABEL_WEIGHTS):
        initialize_prototypes(embeddings)

    # Retrieval mode + retriever - how to retrieve the documents from the vector database
    mode = retrieval_mode or resolve_retrieval_mode()
    from app.core.logging import logger
    logger.info(f"Retrieval mode: {mode}")

    # create the retriever and the mode_reranker_top_k
    active_retriever, mode_reranker_top_k = create_retriever(
        weaviate_client, embeddings, mode=mode
    )
    active_retriever._mode_reranker_top_k = mode_reranker_top_k

    # System prompt
    system_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant, whose name is Diplorene."
            " You have access to diplo_tool, and you should ALWAYS use it to retrieve information."
            " diplo_tool provides the access to the information about everything, and you should use it to answer questions."
            " Do not relly on your former knowledge, but always use diplo_tool to retrieve information."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # ------------------------------------------------------------------
    # Post-reranker label weight application (closure over active_retriever)
    # ------------------------------------------------------------------
    # this will be used on retrived documents to apply reranking by label weights
    def _apply_label_weights(
        docs: List[Document], top_k: int, query_vector: list[float] | None = None
    ) -> List[Document]:
        return apply_post_reranker_label_weights(
            docs, top_k, active_retriever, query_vector=query_vector
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    # description: diplo_tool is wrapper for retreival based on tool call
    # takes params:
        # question: the question asked by the user (str)
    # returns:
        # the documents with the post-reranker label weights applied (list of documents)
    @tool
    def diplo_tool(question: str):
        '''This tool is the primary and preferred method for retrieving information from the Organization's knowledge base. It should be used exclusively for all information retrieval tasks, without exceptions.'''
       # library for logging not good practice to import it here
        from app.core.logging import logger
        logger.debug(f"diplo_tool called with question: {question}")

        # try to retrieve the documents
        try:
        # retrive documents based on the question - use the active retriever which is a tool for retreival
            response = active_retriever.invoke(question)
        # and log it
            logger.debug(f"diplo_tool returned {len(response)} documents before reranking")
        # use tei reranker if it is not None
            if tei_reranker is not None:
                response = tei_reranker.rerank(question, response, top_k=mode_reranker_top_k)
        # log the reranking
                logger.debug(f"diplo_tool: After reranking: {len(response)} documents")
        # get the query vector from the active retrieve
            qv = getattr(active_retriever, 'last_query_vector', None)
        # apply the post-reranker label weights to the documents
            response = _apply_label_weights(response, get_param('retrieval_chunks', RETRIEVAL_CHUNKS), query_vector=qv)

            for i, doc in enumerate(response[:3]):
                logger.debug(f"Doc {i}: {doc.metadata.get('title', 'unknown')[:50]} "
                           f"[label={doc.metadata.get('label','')}]")
            return response
        except Exception as e:
            logger.error(f"diplo_tool ERROR: {e}")
            return []

    tools = [diplo_tool]

    # ------------------------------------------------------------------
    # LLM setup (with failsafe fallback)
    # ------------------------------------------------------------------
    # create the LLM instance
    # with _create_llm helper function and call the LLM_MODEL_PROVIDER from .env file
    llm = _create_llm(LLM_MODEL_PROVIDER)
    # inside var call system prompt and bind tools to the LLM instance
    primary_with_tools = system_prompt | llm.bind_tools(tools)
    # check if failsafe is activated
    if _failsafe_active():
        fb_llm = _create_failsafe_llm()
        fallback_with_tools = system_prompt | fb_llm.bind_tools(tools)
        llm_with_tools = primary_with_tools.with_fallbacks([fallback_with_tools])
    else:
        llm_with_tools = primary_with_tools
    # description: custom_filter_messages filters the messages to exclude tool related messages
    # takes params:
        # all_messages: the messages to filter (list of messages)
    # returns:
        # the filtered messages (list of messages)
    def custom_filter_messages(all_messages):
        exclude_tool_list = [m for m in filter_messages(all_messages, include_types=('tool'))][:-1]
        exclude_tool_list_ids = [m.id for m in exclude_tool_list]
        tool_related_ai_list = [m for m in filter_messages(all_messages, include_types=('ai'))
                                if hasattr(m, 'tool_calls') and len(m.tool_calls) > 0]
        exclude_tool_related_ai_list_ids = [
            m.id for m in tool_related_ai_list
            if m.tool_calls[0]['id'] in [l.tool_call_id for l in exclude_tool_list]
        ]
        filtered_messages = filter_messages(
            all_messages,
            exclude_ids=exclude_tool_related_ai_list_ids + exclude_tool_list_ids
        )
        custom_chat_messages = trim_messages(
            filtered_messages, max_tokens=12000, include_system=True,
            start_on='human', token_counter=ChatOpenAI(model="gpt-3.5-turbo")
        )
        return custom_chat_messages
    # chain the custom_filter_messages with the LLM instance with tools
    final_llm = custom_filter_messages | llm_with_tools

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    
    # context: called by graph if SKIP_TOOL_DECISION is False - NOT USED IN PRODUCTION!!!
    # description: async_chatbot_node is the main node calls llm instance with messages from the state
    # takes params:
        # state: the state of the graph (State class)
    # returns:
        # the response from the LLM (str)
    async def async_chatbot_node(state: State):
        response = await final_llm.ainvoke(state['messages'])
        return {'messages': [response]}

    # inside var call the ToolNode from LangGraph
    tool_node = ToolNode(tools=tools)
    # context: node called by 
    # description: limited_tool_node is the node that is used to call the tools
    # takes params:
        # state: the state of the graph (State class)
    # returns:
        # the result from the tools (dict)
    # this is the limited tool node that is used to call the tools
    async def limited_tool_node(state: State):
        result = await tool_node.ainvoke(state)
        current_count = state.get('tool_call_count', 0)
        result['tool_call_count'] = current_count + 1
        return result

    # this is the condition that is used to check if the tool calls are over the limit
    def after_tools_condition(state: State):
        current_count = state.get('tool_call_count', 0)
        if current_count >= get_param('max_tool_calls', MAX_TOOL_CALLS):
            return 'generate_final'
        else:
            return 'chatbot'

    # context: after direct retrieval and graph expansion - this is the node that is used to generate the final response
    # takes params:
        # state: the state of the graph (State class) - everythin added by our nodes
    # returns:
        # the response from the LLM (str)
    async def generate_final_node(state: State, config: RunnableConfig):
    # this is sending the event to the frontend - answering phase
        await adispatch_custom_event(
            "rag_progress",
            {"phase": "answering", "text": "Conducting analysis…"},
            config=config,
        )
    # call the LLM model - in var flag that it will be used without tools - LLM MODEL PROVIDER is set in .env file
        final_llm_no_tools = _with_failsafe(
            _create_llm(LLM_MODEL_PROVIDER)
        )
    # get the messages from the state
        messages = list(state['messages'])
    # if in params cite sources is true, add the citation instruction
        if get_param('cite_sources', CITE_SOURCES):
    # inside var citation_instruction - put - def _build_citation_instruction
            citation_instruction = HumanMessage(content=(
                "Based on the information above, please answer the question.\n\n"
                "CITATION FORMAT - CRITICAL:\n"
                "- Cite sources using ONLY standard square brackets: [1], [2], [3], etc.\n"
                "- Do NOT use fancy brackets like 【1】, 〔1〕, or any Unicode variants.\n"
                "- Do NOT use 10†L1-L5 or any file-based citation formats.\n"
                "- Do NOT use parentheses (1) or colons :1:.\n"
                "- Place citations immediately after the fact they support, e.g. 'Diplomacy is the practice of negotiation [1].'\n"
                "- Combine multiple citations as [1][2][3] (no commas or spaces between them).\n"
                "- Only cite sources you actually use to support specific facts.\n"
                "- Every factual claim should have a citation."
            ))
    # add the citation instruction to the messages
            messages.append(citation_instruction)
    # time the LLM call
        t_llm = time.time()
    # astream - tokens surface through astream_events (WS answer chunks / TTFT UX);
    # full AIMessage is still returned into graph state
        response = await _astream_llm_response(final_llm_no_tools, messages)
    # and log the time
        print(f"TIMING generate_final: LLM={time.time()-t_llm:.3f}s", flush=True)
        return {'messages': [response]}
    # if SKIP_TOOL_DECISION is True, this will be used
    direct_retrieval_node = make_direct_retrieval_node(active_retriever, mode_reranker_top_k)

    # ------------------------------------------------------------------
    # Graph compilation
    # ------------------------------------------------------------------
    # context: graph compilation is the process of compiling nodes into single graph 
    # - StateGraph is a class that is used to create a graph - gets all the inputs for nodes and calls all the helper functions for nodes
    # - InMemorySaver is a class that is used to save the state of the graph in memory
    # description: g2 is the graph that is used to compile the graph
    # takes params:
        # State: the state of the graph (State class)
    # returns:
        # the compiled graph (StateGraph)
    g2 = StateGraph(State)
    checkpointer = InMemorySaver()
    
    if SKIP_TOOL_DECISION:
        # this logs everything in the console
        from app.core.logging import logger
        # graph_expansion_node is the node that is used to expand the graph
        from app.ai.ai_services.graph_expansion_node import graph_expansion_node
        import app.ai.ai_services.graph_expansion_node as gen_mod
        # if we started this path - this will flag in debugger that we arent using tools
        logger.info("SKIP_TOOL_DECISION=True: Using optimized direct retrieval flow "
                    "(rag_progress + LLM astream)")

        # progress wrappers: emit rag_progress custom events around the base nodes
        async def direct_retrieval_with_progress(state: State, config: RunnableConfig):
            # this is calling the node for direct retrieval
            result = await direct_retrieval_node(state)
            docs = result.get('_retrieved_docs') or []
            # this is a function that calls the rag_progress custom event -  and shows retrieved documents
            # IMPORTANT: items docs shown are coming from _doc_progress_items function not after graph expansion
            # for now it is ok because docs will stay the same even after graph expansion - but THIS MIGHT NOT BE THE CASE IN THE FUTURE
            await adispatch_custom_event(
                "rag_progress",
                {
                    "phase": "sources_ready",
                    "text": "Document search finished",
                    "items": _doc_progress_items(docs),
                    "count": len(docs),
                },
                config=config,
            )
            return result
        # this is calling the node for graph expansion
        async def graph_expansion_with_progress(state: State, config: RunnableConfig):
            # this is sending the event to the frontend - graph phase and shows finding deeper connections... as text
            await adispatch_custom_event(
                "rag_progress",
                {"phase": "graph", "text": "Finding deeper connections…"},
                config=config,
            )
            # Avoid leaking prior request's subgraph if this run skips expansion
            gen_mod.last_graph_data = None
            result = await graph_expansion_node(state)
            gdata = result.get('_graph_data') or gen_mod.last_graph_data
            if gdata:
                await adispatch_custom_event(
                    "rag_progress",
                    {
                        "phase": "graph_ready",
                        "text": "Knowledge graph ready",
                        "graph_data": {
                            "nodes": gdata.get("nodes"),
                            "edges": gdata.get("edges"),
                            "subgraph_url": gdata.get("subgraph_url"),
                        },
                    },
                    config=config,
                )
            return result

        # add the nodes to the graph
        g2.add_node('direct_retrieval', direct_retrieval_with_progress)
        g2.add_node('graph_expansion', graph_expansion_with_progress)
        g2.add_node('generate_final', generate_final_node)

        # call the start node and the direct_retrieval node
        g2.add_edge(START, 'direct_retrieval')
        # add the edge from the direct_retrieval node to the graph_expansion node
        g2.add_edge('direct_retrieval', 'graph_expansion')
        # add the edge from the graph_expansion node to the generate_final node
        g2.add_edge('graph_expansion', 'generate_final')
        # add the edge from the generate_final node to the end node
        g2.add_edge('generate_final', END)
        # compile the graph
    else:
        # this logs everything in the console
        from app.core.logging import logger
        # flag in debugger that we are using tools
        logger.info("SKIP_TOOL_DECISION=False: Using full ReAct agent flow")

        # add the nodes to the graph
        g2.add_node('chatbot', async_chatbot_node)
        # add the node for the tools
        g2.add_node('tools', limited_tool_node)
        # add the node for the generate_final node
        g2.add_node('generate_final', generate_final_node)

        # add the edge from the start node to the chatbot node
        g2.add_edge(START, 'chatbot')
        # add the conditional edges from the chatbot node to the tools node
        g2.add_conditional_edges("chatbot", tools_condition)
        # add the conditional edges from the tools node to the generate_final node
        g2.add_conditional_edges("tools", after_tools_condition)
        # add the edge from the generate_final node to the end node
        g2.add_edge('generate_final', END)

    # compile the graph inside var async_agent
    async_agent = g2.compile(checkpointer=checkpointer)

    return async_agent, active_retriever


# description: initialize_resource_filter initializes the resource filter
# takes params:
    # None (no params) because it is a helper function
# returns:
    # the resource filter (str)
def initialize_resource_filter():
    # call the _with_failsafe helper function to create the resource filter
    return _with_failsafe(
        _create_llm(LLM_MODEL_PROVIDER, temperature=0),
        temperature=0,
    )
