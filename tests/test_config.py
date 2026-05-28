from fbneo.config import load_settings


MODEL_ENV_KEYS = [
    "EMBEDDING_PROVIDER",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "GEMINI_API_KEY",
]


def _clear_model_env(monkeypatch) -> None:
    for key in MODEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_default_models_target_openrouter_qwen_and_gemini(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_model_env(monkeypatch)

    settings = load_settings()

    assert settings.embedding_provider == "openrouter"
    assert settings.embedding_base_url == "https://openrouter.ai/api/v1"
    assert settings.embedding_model == "qwen/qwen3-embedding-4b"
    assert settings.embedding_dimension == 2560
    assert settings.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert settings.llm_model == "gemini-2.5-flash"


def test_provider_specific_api_key_fallbacks(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    settings = load_settings()

    assert settings.embedding_api_key == "openrouter-test-key"
    assert settings.llm_api_key == "gemini-test-key"


def test_hash_embedding_defaults_remain_available(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")

    settings = load_settings()

    assert settings.embedding_provider == "hash"
    assert settings.embedding_dimension == 384
    assert settings.embedding_model == "hash"
    assert settings.embedding_base_url == ""


def test_openai_embedding_defaults_remain_supported(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")

    settings = load_settings()

    assert settings.embedding_provider == "openai"
    assert settings.embedding_dimension == 1536
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_base_url == "https://api.openai.com/v1"
