"""Debug logging for LLM model selection."""


def log_model_configured(provider: str, model: str) -> None:
    print(f"DEBUG: llm configured provider={provider} model={model}", flush=True)
