import logging
import os
from typing import List

from databases import DatabaseURL
from loguru import logger
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings, Secret

from app.core.logging import InterceptHandler

API_PREFIX = "/api"

JWT_TOKEN_PREFIX = "Token"  # noqa: S105 在做用户校验时，需要把这个前缀加上，并加空格
VERSION = "0.1.0"

def _load_config() -> Config:
    """Load .env from disk when present; otherwise use process env (Docker env_file:)."""
    env_file = os.getenv("ENV_FILE", ".env")
    if env_file and os.path.isfile(env_file):
        return Config(env_file)
    return Config()


config = _load_config()
global_config = config

DEBUG: bool = config("DEBUG", cast=bool, default=False)

WEBSITE_NAME:str = config("WEBSITE_NAME",cast=str)
HOST:str = config("HOST",cast=str,default="127.0.0.1")
PORT:int = config("PORT",cast=int,default=8560)
RELOAD:bool = config("RELOAD",cast=bool,default=True)

DATABASE_URL: DatabaseURL = config("DB_CONNECTION", cast=DatabaseURL)
MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)
DB_HOST:str = config("DB_HOST",cast=str,default="127.0.0.1")
DB_PORT:int = config("DB_PORT",cast=int,default=3306)
DB_USER:str = config("DB_USER",cast=str)
DB_PWD:str = config("DB_PWD",cast=str,default="")
DB_NAME:str = config("DB_NAME",cast=str,default="")

INDEX_NAME:str = config("INDEX_NAME",cast=str)
INDEX_NAME2:str = config("INDEX_NAME2",cast=str)
LLM_MODEL_NAME:str = config("LLM_MODEL_NAME",cast=str, default="gpt-4o")
LLM_MODEL_PROVIDER:str = config("LLM_MODEL_PROVIDER",cast=str, default="openai")
EMBEDDING_MODEL_NAME:str = config("EMBEDDING_MODEL_NAME",cast=str, default="text-embedding-3-large")

# Embedding provider configuration
EMBEDDING_PROVIDER:str = config("EMBEDDING_PROVIDER", cast=str, default="openai")
LOCAL_EMBEDDING_URL:str = config("LOCAL_EMBEDDING_URL", cast=str, default="")
LOCAL_EMBEDDING_KEY:str = config("LOCAL_EMBEDDING_KEY", cast=str, default="")

# Local LLM configuration (used when LLM_MODEL_PROVIDER = 'local')
LOCAL_LLM_URL:str = config("LOCAL_LLM_URL", cast=str, default="")
LOCAL_LLM_MODEL:str = config("LOCAL_LLM_MODEL", cast=str, default="")
LOCAL_LLM_KEY:str = config("LOCAL_LLM_KEY", cast=str, default="")

# DeepSeek LLM configuration (used when LLM_MODEL_PROVIDER = 'deepseek')
# OpenAI-compatible API; model id comes from LLM_MODEL_NAME (same as openai/anthropic).
DEEPSEEK_API_KEY:str = config("DEEPSEEK_API_KEY", cast=str, default="")
DEEPSEEK_API_URL:str = config(
    "DEEPSEEK_API_URL", cast=str, default="https://api.deepseek.com/v1"
)

# LLM Failsafe: automatic fallback to a secondary LLM when the primary is unreachable
# When enabled and primary LLM call fails (timeout, connection error, etc.),
# the system retries with the failsafe LLM provider.
LLM_FAILSAFE_ENABLED:bool = config("LLM_FAILSAFE_ENABLED", cast=bool, default=True)
FAILSAFE_LLM_PROVIDER:str = config("FAILSAFE_LLM_PROVIDER", cast=str, default="openai")
FAILSAFE_LLM_MODEL:str = config("FAILSAFE_LLM_MODEL", cast=str, default="")
FAILSAFE_LLM_URL:str = config("FAILSAFE_LLM_URL", cast=str, default="")
FAILSAFE_LLM_KEY:str = config("FAILSAFE_LLM_KEY", cast=str, default="")

# Source filtering option - set to False to skip additional LLM call for filtering sources
USE_SOURCE_FILTER:bool = config("USE_SOURCE_FILTER", cast=bool, default=True)

# Maximum number of tool calls allowed per request
# LLM uses ReAct pattern where it can call tools multiple times in a loop.
# Sometimes LLM hallucinates unnecessary additional tool calls, wasting tokens and time.
# Set this to 1 to force single tool call (recommended for simple Q&A).
# Set higher (2-3) if you need multi-step reasoning (e.g., comparing multiple topics).
MAX_TOOL_CALLS:int = config("MAX_TOOL_CALLS", cast=int, default=1)

# Skip LLM tool decision - directly call retrieval without asking LLM which tool to use
# When True: Skips the first LLM call that decides to use diplo_tool (saves ~1s)
# When False: Uses full ReAct agent pattern where LLM decides which tool to call
# Set to True if you only have one tool (diplo_tool) that should always be used.
# Set to False if you have multiple tools and need LLM to choose between them.
SKIP_TOOL_DECISION:bool = config("SKIP_TOOL_DECISION", cast=bool, default=False)

# Number of chunks to retrieve and send to LLM as context
# Higher = more context, potentially better answers, but slower (~0.5s per 2 chunks)
# Lower = faster responses, less context
# Recommended: 6-8 for balanced speed/quality
RETRIEVAL_CHUNKS:int = config("RETRIEVAL_CHUNKS", cast=int, default=8)

# Hybrid Search Alpha (balance between vector and BM25)
# alpha=1.0: 100% vector search (pure semantic)
# alpha=0.5: 50% vector + 50% BM25 (balanced)
# alpha=0.0: 100% BM25 (pure keyword)
# Recommended: 0.85 (85% vector + 15% BM25) for semantic-dominant search
HYBRID_ALPHA:float = config("HYBRID_ALPHA", cast=float, default=0.85)

# Cite sources in answer - LLM will cite sources as [1], [2], etc.
# When True: Only sources actually cited in the answer are returned
# When False: All retrieved sources are returned
# Works with streaming, no extra latency
CITE_SOURCES:bool = config("CITE_SOURCES", cast=bool, default=True)

# When True, the API accepts an optional system_prompt field from the frontend request.
# When False, the hardcoded prompt from app/ai/prompts/ is always used.
USE_CUSTOM_SYSTEM_PROMPT:bool = config("USE_CUSTOM_SYSTEM_PROMPT", cast=bool, default=False)

# Retrieval Mode: 'sentence', 'paragraph', 'combined', or 'twophase'
# 'sentence'  = Sentence-first retrieval via DiploChunk with aggregation (original)
# 'paragraph' = Direct paragraph-level retrieval via DiploParagraph
# 'combined'  = Both sentence + paragraph search, merged & deduplicated before reranking
# 'twophase'  = Pure vector search + title filter → section selection → reranker (no BM25)
RETRIEVAL_MODE:str = config("RETRIEVAL_MODE", cast=str, default="sentence")

# Sentence-First Retrieval - search sentences first, then aggregate to sections
# When True: Uses DiploChunk (sentences) with aggregation for better precision
# When False: Uses paragraph-level retrieval (current behavior)
# DEPRECATED: Use RETRIEVAL_MODE instead. Kept for backward compatibility.
USE_SENTENCE_RETRIEVAL:bool = config("USE_SENTENCE_RETRIEVAL", cast=bool, default=False)

# Number of sentences to retrieve in sentence-first mode (before aggregation)
# More sentences = better coverage but slower. Recommended: 150-300
SENTENCE_RETRIEVAL_K:int = config("SENTENCE_RETRIEVAL_K", cast=int, default=200)

# A/B path: sentence_header (parallel h1 titles + sentences) — used by /api/chat-sh only
# Does not change default RETRIEVAL_MODE / /api/chat
SENTENCE_HEADER_RETRIEVAL_K: int = config("SENTENCE_HEADER_RETRIEVAL_K", cast=int, default=50)
SENTENCE_TITLE_K: int = config("SENTENCE_TITLE_K", cast=int, default=20)
SENTENCE_TITLE_FETCH_LIMIT: int = config("SENTENCE_TITLE_FETCH_LIMIT", cast=int, default=60)
SENTENCE_TITLE_ALPHA: float = config("SENTENCE_TITLE_ALPHA", cast=float, default=0.3)

# Sentence Blocklist — filter out boilerplate/duplicate sentences from retrieval
USE_SENTENCE_BLOCKLIST:bool = config("USE_SENTENCE_BLOCKLIST", cast=bool, default=True)

# Number of paragraphs to retrieve in paragraph or combined mode
# In combined mode these are merged with sentence results before reranking
PARAGRAPH_RETRIEVAL_K:int = config("PARAGRAPH_RETRIEVAL_K", cast=int, default=20)

# Force-include configuration
# H1: match query terms against page title (h1)
# SECTION: match query terms against section headings (last_h_title: h2-h6/strong)
FORCE_INCLUDE_H1:bool = config("FORCE_INCLUDE_H1", cast=bool, default=True)
FORCE_INCLUDE_SECTION:bool = config("FORCE_INCLUDE_SECTION", cast=bool, default=False)
FORCE_INCLUDE_OVERLAP:float = config("FORCE_INCLUDE_OVERLAP", cast=float, default=0.6)
FORCE_INCLUDE_MAX:int = config("FORCE_INCLUDE_MAX", cast=int, default=7)

# Raw paragraphs to fetch from hybrid search before URL aggregation
PARAGRAPH_FETCH_LIMIT:int = config("PARAGRAPH_FETCH_LIMIT", cast=int, default=500)

# TwoPhase: number of URLs to pass from Phase 1 to Phase 2 (independent of RERANKER_TOP_K)
TWOPHASE_K_URLS:int = config("TWOPHASE_K_URLS", cast=int, default=40)

# TwoPhase Phase 1b: heading boost weight for last_h_title similarity
# 0.0 = disabled, 0.5 = moderate (recommended), 1.0 = aggressive
HEADING_BOOST_WEIGHT:float = config("HEADING_BOOST_WEIGHT", cast=float, default=0.5)

# Sentence retrieval collection name
SENTENCE_INDEX_NAME:str = config("SENTENCE_INDEX_NAME", cast=str, default="Sentences")

# Use contextual collections (_contextual suffix) for enriched vector search
USE_CONTEXTUAL_COLLECTIONS:bool = config("USE_CONTEXTUAL_COLLECTIONS", cast=bool, default=False)

# Reranker configuration
USE_RERANKER:bool = config("USE_RERANKER", cast=bool, default=False)
RERANKER_URL:str = config("RERANKER_URL", cast=str, default="")
RERANKER_API_KEY:str = config("RERANKER_API_KEY", cast=str, default="")
RERANKER_TOP_K:int = config("RERANKER_TOP_K", cast=int, default=20)

# Mode-specific reranker candidate counts (0 = use global RERANKER_TOP_K)
TWOPHASE_RERANKER_TOP_K:int = config("TWOPHASE_RERANKER_TOP_K", cast=int, default=0)
SENTENCE_RERANKER_TOP_K:int = config("SENTENCE_RERANKER_TOP_K", cast=int, default=0)

# Sentence-level cross-encoder reranking (Step 3b)
# Selects the best sentence per group for deep link highlighting.
SENTENCE_CE_CANDIDATES:int = config("SENTENCE_CE_CANDIDATES", cast=int, default=3)
SENTENCE_CE_MAX_GROUPS:int = config("SENTENCE_CE_MAX_GROUPS", cast=int, default=20)

# Sentence highlight mode for deep links:
#   single = cross-encoder picks one best sentence per section (Step 3b)
#   multi  = all matched sentences highlighted (Step 3b skipped)
SENTENCE_HIGHLIGHT_MODE:str = config("SENTENCE_HIGHLIGHT_MODE", cast=str, default="single")

# Max sentences to include in multi-highlight deep links (sorted by score desc).
# 0 = no limit. Recommended: 5
MULTI_HIGHLIGHT_MAX_SENTS:int = config("MULTI_HIGHLIGHT_MAX_SENTS", cast=int, default=5)

# Query-level retrieval cache TTL in seconds (0 = disabled)
# Caches post-reranker docs in Redis to skip embedding + Weaviate + reranker on repeat queries
RETRIEVAL_CACHE_TTL:int = config("RETRIEVAL_CACHE_TTL", cast=int, default=120)

# Deep link URL shortening via Redis
USE_SHORT_DEEP_LINKS:bool = config("USE_SHORT_DEEP_LINKS", cast=bool, default=True)
REDIS_HOST:str = config("REDIS_HOST", cast=str, default="localhost")
REDIS_PORT:int = config("REDIS_PORT", cast=int, default=6379)
REDIS_DB:int = config("REDIS_DB", cast=int, default=0)
DEEP_LINK_TTL_DAYS:int = config("DEEP_LINK_TTL_DAYS", cast=int, default=30)
DEEP_LINK_API_URL:str = config("DEEP_LINK_API_URL", cast=str, default="")

# Max sections per URL sent to reranker (sentence/combined mode)
# Prevents one page with many sections from monopolizing reranker slots.
# 0 = no limit. Recommended: 3
MAX_SECTIONS_PER_URL:int = config("MAX_SECTIONS_PER_URL", cast=int, default=0)

# Max sentences used for scoring per section (sentence/combined mode)
# Only the top N highest-scoring sentences contribute to aggregation.
# Prevents sections with many sentences from inflating scores via quantity.
# 0 = no limit. Recommended: 5
TOP_N_PER_SECTION:int = config("TOP_N_PER_SECTION", cast=int, default=0)

# Legacy heading boost/injection (unused; contextual retrieval only)
# but underrepresented in hybrid results. Improves recall for topic pages.
HEADING_INJECT_MIN_SIM:float = config("HEADING_INJECT_MIN_SIM", cast=float, default=0.65)
HEADING_INJECT_MAX:int = config("HEADING_INJECT_MAX", cast=int, default=10)
HEADING_INJECT_MIN_SENTS:int = config("HEADING_INJECT_MIN_SENTS", cast=int, default=5)

# Dynamic Label Weights — boost label weights based on query-prototype embedding similarity
# When True, computes cosine similarity between query and post_type prototypes at query time.
# Dynamic boost can only lift weights above static baseline (max formula), never reduce.
USE_DYNAMIC_LABEL_WEIGHTS:bool = config("USE_DYNAMIC_LABEL_WEIGHTS", cast=bool, default=True)
DYNAMIC_WEIGHT_ALPHA:float = config("DYNAMIC_WEIGHT_ALPHA", cast=float, default=2.5)
DYNAMIC_WEIGHT_MIN_SIM:float = config("DYNAMIC_WEIGHT_MIN_SIM", cast=float, default=0.3)

# Optional recency boost after reranker (off unless retrieval_config.use_recency_boost=true)
USE_RECENCY_BOOST:bool = config("USE_RECENCY_BOOST", cast=bool, default=False)
RECENCY_BOOST_MAX_AGE_DAYS:int = config("RECENCY_BOOST_MAX_AGE_DAYS", cast=int, default=1825)
RECENCY_HALF_LIFE_DAYS:int = config("RECENCY_HALF_LIFE_DAYS", cast=int, default=900)
RECENCY_MAX_BOOST:float = config("RECENCY_MAX_BOOST", cast=float, default=0.35)

# Deep Link Highlight Mode: 'sentence' or 'paragraph'
# 'sentence' = highlight only the best matching sentence (more precise)
# 'paragraph' = highlight the entire paragraph/section (more context)
DEEP_LINK_HIGHLIGHT_MODE:str = config("DEEP_LINK_HIGHLIGHT_MODE", cast=str, default="paragraph")

# PDF Proxy (chatbot-via) for PDF highlighting in the viewer
# When set, PDF sources get routed through this proxy with hash-based highlight params
# Example: https://chatbot-via.diplomacy.edu
PDF_PROXY_BASE_URL:str = config("PDF_PROXY_BASE_URL", cast=str, default="")

# Viewer for ingested documents without a public source_url (uploaded PDFs, dumps).
# Public base URL of this API, e.g. https://mydepartment-api-dev.diplomacy.edu
# Empty = feature off (such sources stay unlinked).
DOC_VIEWER_BASE_URL:str = config("DOC_VIEWER_BASE_URL", cast=str, default="")

# Hosts that serve the Diplo highlight script and therefore understand
# ?diplo-deep-link-text=. Third-party pages get a browser-native #:~:text=
# fragment instead, since our script is not there to read the query param.
HIGHLIGHT_SCRIPT_HOSTS:list = config(
    "HIGHLIGHT_SCRIPT_HOSTS",
    cast=lambda v: [h.strip().lower() for h in v.split(",") if h.strip()],
    default="diplomacy.edu",
)

# Related Questions Model Configuration
# Provider: 'openai', 'local', or 'deepseek'
RELATED_QUESTIONS_PROVIDER:str = config("RELATED_QUESTIONS_PROVIDER", cast=str, default="openai")

# Local model settings (used when provider = 'local')
RELATED_QUESTIONS_LOCAL_URL:str = config("RELATED_QUESTIONS_LOCAL_URL", cast=str, default="")
RELATED_QUESTIONS_LOCAL_MODEL:str = config("RELATED_QUESTIONS_LOCAL_MODEL", cast=str, default="")
RELATED_QUESTIONS_LOCAL_KEY:str = config("RELATED_QUESTIONS_LOCAL_KEY", cast=str, default="")

# OpenAI model for related questions (used when provider = 'openai')
RELATED_QUESTIONS_OPENAI_MODEL:str = config("RELATED_QUESTIONS_OPENAI_MODEL", cast=str, default="gpt-4o-mini")

# DeepSeek model for related questions (used when provider = 'deepseek')
RELATED_QUESTIONS_DEEPSEEK_MODEL:str = config(
    "RELATED_QUESTIONS_DEEPSEEK_MODEL", cast=str, default="deepseek-v4-flash"
)

OPENAI_KEY:str = global_config("OPENAI_KEY",cast=str)
OPENAI_ORGANIZATION:str = global_config("OPENAI_ORGANIZATION",cast=str)
WV_KEY:str = global_config("WV_KEY",cast=str)
WV_CLIENT_URL:str = config("WV_CLIENT_URL",cast=str)
# Required per environment (no auto-detect magics in graph/ingest).
WV_GRPC_PORT:int = config("WV_GRPC_PORT", cast=int, default=0)

# Default Weaviate access_groups filter when request omits access_group_name(s).
# Comma-separated, e.g. diplo_team or diplo_team,ai_team. Empty = no ACL filter.
DEFAULT_ACCESS_GROUPS: List[str] = config(
    "DEFAULT_ACCESS_GROUPS", cast=CommaSeparatedStrings, default="diplo_team"
)

# When True, exclude objects with copyright=1 from retrieval (0 = free to use).
EXCLUDE_COPYRIGHT: bool = config("EXCLUDE_COPYRIGHT", cast=bool, default=True)

SECRET_KEY: Secret = config("SECRET_KEY", cast=Secret)

# Langfuse observability (self-hosted at langfuse.diplomacy.edu)
LANGFUSE_ENABLED: bool = config("LANGFUSE_ENABLED", cast=bool, default=False)
LANGFUSE_PUBLIC_KEY: str = config("LANGFUSE_PUBLIC_KEY", cast=str, default="")
LANGFUSE_SECRET_KEY: str = config("LANGFUSE_SECRET_KEY", cast=str, default="")
LANGFUSE_HOST: str = config(
    "LANGFUSE_HOST", cast=str, default="https://langfuse.diplomacy.edu"
)
LANGFUSE_DEBUG: bool = config("LANGFUSE_DEBUG", cast=bool, default=False)

PROJECT_NAME: str = config("PROJECT_NAME", default="FastAPI example application")
ALLOWED_HOSTS: List[str] = config(
    "ALLOWED_HOSTS", cast=CommaSeparatedStrings, default=""
)
ALLOWED_IPS: List[str] = config(
    "ALLOWED_IPS", cast=CommaSeparatedStrings, default=""
)

PERSON_LIST_CACHE_TTL_SECONDS: int = config(
    "PERSON_LIST_CACHE_TTL_SECONDS", cast=int, default=3600
)

# MyDepartment chatbot gateway (/api/chatbot)
# A chatbot is a Chatbot Generator assistant. Its prompt, knowledge scope and
# org-scoped retrieval parameters are assembled by the Department backend on
# every run, so the gateway drives that backend instead of LangGraph directly.
DEPARTMENT_API_BASE_URL: str = config(
    "DEPARTMENT_API_BASE_URL", cast=str, default="https://api.mydepartment.ai"
)
CHATBOT_GENERATOR_API_PATH: str = config(
    "CHATBOT_GENERATOR_API_PATH", cast=str, default="/api/chatbot-generator"
)
# When set, /api/chatbot/* is reachable from any IP that presents this key and
# the IP allowlist no longer applies to it (see IPAllowlistMiddleware).
CHATBOT_INVOKE_API_KEY: str = config("CHATBOT_INVOKE_API_KEY", cast=str, default="")
# Budget for a whole chatbot turn (retrieval + generation), not per chunk.
CHATBOT_INVOKE_TIMEOUT: float = config(
    "CHATBOT_INVOKE_TIMEOUT", cast=float, default=300.0
)
# The chatbot catalog is not an embed route, so unlike the run endpoints it
# needs the Department backend's own service key to answer.
DEPARTMENT_API_KEY: str = config("DEPARTMENT_API_KEY", cast=str, default="")
DEPARTMENT_API_KEY_HEADER: str = config(
    "DEPARTMENT_API_KEY_HEADER", cast=str, default="diplo-sso-api-key"
)

# logging configuration

LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO

# Configure Loguru to write to debug.log
logger.remove()  # Remove default handler to avoid duplicate logs
logger.add('debug.log', level=LOGGING_LEVEL, rotation="1024 MB", retention="30 days")
