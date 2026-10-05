"""Configuration loading and environment resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class DatasetCfg:
    source: str = "hf"
    name: str = "universal-dependencies/universal_dependencies"
    config: str = "ja_gsd"
    split: str = "test"
    revision: str | None = None
    path: str | None = None
    ids_path: str | None = None  # split id list (KWDLC / KNP)


@dataclass
class ModelCfg:
    name: str
    model: str | None = None  # API model id, if different from the display name
    base_url: str | None = None
    base_url_env: str | None = None
    native_base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api: str = "openai"  # "openai" (OpenAI-compatible) or "ollama" (native, with timings)
    reasoning_effort: str | None = None  # none|low|medium|high
    num_ctx: int | None = None  # Ollama context window override
    max_tokens: int | None = None  # per-model output cap (falls back to llm.max_tokens)
    extra_body: dict | None = None  # provider-specific request body extras
    extra_headers: dict | None = None  # provider-specific HTTP headers
    requires_api_key: bool = False  # skip when the key env var is absent
    tier: str = "local"
    enabled: bool = True
    disable_tools: bool = True

    def api_model(self) -> str:
        """Model id sent to the API (falls back to the display name)."""
        return self.model or self.name

    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        if self.base_url_env:
            return os.environ.get(self.base_url_env) or None
        return None

    def resolved_native_base_url(self) -> str | None:
        """Root URL for Ollama's native API (strip a trailing /v1)."""
        url = self.native_base_url or self.resolved_base_url()
        if not url:
            return None
        url = url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        return url

    def resolved_api_key(self) -> str:
        return os.environ.get(self.api_key_env) or "not-needed"

    def has_api_key(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def usable(self) -> bool:
        """Resolvable base_url and, if required, a present API key."""
        if self.resolved_base_url() is None:
            return False
        if self.requires_api_key and not self.has_api_key():
            return False
        return True

    def available(self) -> bool:
        """A model is usable if enabled and its base_url can be resolved."""
        return self.enabled and self.resolved_base_url() is not None


@dataclass
class LLMCfg:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    retries: int = 1
    lemmas: bool = False
    reasoning_effort: str | None = None  # global default; models may override
    batch_size: int | None = None  # None = single-sentence reference prompt


@dataclass
class Config:
    root: Path
    dataset: DatasetCfg
    models: list[ModelCfg]
    llm: LLMCfg = field(default_factory=LLMCfg)
    gold_dir: str = "data/gold"
    results_dir: str = "results"
    prompt: str = "prompts/segment_ja.txt"
    prompt_lemma: str = "prompts/segment_ja_lemma.txt"
    prompt_batch: str = "prompts/segment_ja_batch.txt"
    prompt_batch_lemma: str = "prompts/segment_ja_batch_lemma.txt"
    seed: int = 1234
    bootstrap: int = 1000

    @property
    def treebank(self) -> str:
        return self.dataset.config

    def gold_path(self) -> Path:
        return self.root / self.gold_dir / self.treebank / "gold.jsonl"

    def raw_path(self, model: str) -> Path:
        return self.root / self.results_dir / "raw" / self.treebank / f"{_slug(model)}.jsonl"

    def predictions_path(self, analyzer: str) -> Path:
        return (
            self.root
            / self.results_dir
            / "predictions"
            / self.treebank
            / f"{_slug(analyzer)}.jsonl"
        )

    def prompt_path(self, lemma: bool = False, batch: bool = False) -> Path:
        if batch:
            return self.root / (self.prompt_batch_lemma if lemma else self.prompt_batch)
        return self.root / (self.prompt_lemma if lemma else self.prompt)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (lists/scalars replace)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    root = path.parent
    load_dotenv(root / ".env")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    base = data.pop("extends", None)
    if base:
        base_data = yaml.safe_load((path.parent / base).read_text(encoding="utf-8")) or {}
        data = _deep_merge(base_data, data)

    dataset = DatasetCfg(**{**data.get("dataset", {})})
    llm = LLMCfg(**{**data.get("llm", {})})
    models = [ModelCfg(**m) for m in data.get("models", [])]

    return Config(
        root=root,
        dataset=dataset,
        models=models,
        llm=llm,
        gold_dir=data.get("gold_dir", "data/gold"),
        results_dir=data.get("results_dir", "results"),
        prompt=data.get("prompt", "prompts/segment_ja.txt"),
        prompt_lemma=data.get("prompt_lemma", "prompts/segment_ja_lemma.txt"),
        prompt_batch=data.get("prompt_batch", "prompts/segment_ja_batch.txt"),
        prompt_batch_lemma=data.get(
            "prompt_batch_lemma", "prompts/segment_ja_batch_lemma.txt"
        ),
        seed=int(data.get("seed", 1234)),
        bootstrap=int(data.get("bootstrap", 1000)),
    )
