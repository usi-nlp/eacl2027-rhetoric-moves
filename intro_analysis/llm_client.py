#!/usr/bin/env python3
"""
Unified, multi-provider LLM client with prompt caching for the intro-analysis
experiments.

`run_experiment.py` (full-intro segmentation/classification) sends the SAME
large, static system prompt (the CARS taxonomy / full annotation manual, plus the
few-shot worked examples) on every call and varies only the short per-task user
message. That is the ideal shape for **prompt caching**: the expensive prefix is
billed once and re-read at a steep discount on every subsequent call.

This module exposes a single `LLMClient` that talks to three back-ends and turns
caching on for each:

  * **OpenAI**  (`gpt-*`, `o*`) — automatic prefix caching for prompts >1024
    tokens; we additionally pass a stable `prompt_cache_key` so all calls in a run
    route to the same cache.
  * **Anthropic** (`claude-*`) — explicit caching via a `cache_control:
    {"type": "ephemeral"}` breakpoint on the system block.
  * **Google Gemini** (`gemini-*`) — implicit caching: Gemini 2.5 models cache
    automatically when a request repeats a large prefix, so sending the same
    `system_instruction` on every call gets the cached rate with no extra code.

Caching is a provider feature — we only structure each request so the API can
apply it (a stable, identical system prompt as the cached prefix); we don't
manage caches ourselves.

The provider is inferred from the model name (override with `provider=`). Every
back-end returns plain text; use `parse_json()` to decode the model's JSON reply
(tolerates ```json fences that Claude/Gemini sometimes add).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------

def infer_provider(model: str) -> str:
    """Map a model name to its provider."""
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    # gpt-*, o1/o3/o4, chatgpt-*, …
    return "openai"


# Reasoning-effort → thinking-budget (tokens) for providers that take a numeric
# budget rather than an "effort" enum. low keeps calls cheap.
_THINKING_BUDGET = {"low": 0, "medium": 4096, "high": 16384}


# ---------------------------------------------------------------------------
# JSON parsing helper (shared by callers)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json(text: str):
    """Parse a model reply as JSON, tolerating markdown code fences.

    Returns the decoded object, or raises json.JSONDecodeError if it cannot be
    recovered. Callers should handle the exception the same way they handled a
    raw json.loads failure before.
    """
    if text is None:
        raise json.JSONDecodeError("empty model output", "", 0)
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


# ---------------------------------------------------------------------------
# Unified client
# ---------------------------------------------------------------------------

class LLMClient:
    """Provider-agnostic text generator with prompt caching enabled.

    Parameters
    ----------
    model:
        The model name (e.g. "gpt-5.5", "claude-sonnet-4-6", "gemini-2.5-pro").
    provider:
        "auto" (default — inferred from the model name) or one of
        "openai" / "anthropic" / "google".
    reasoning_effort:
        "low" | "medium" | "high". Passed through to OpenAI/Gemini reasoning
        controls; for Anthropic it toggles extended thinking (low = off).
    max_output_tokens:
        Generation cap. Required by Anthropic/Gemini; left high so the verbatim
        segment output is never truncated.
    use_cache:
        Enable prompt caching for the static system prompt (default True).
    anthropic_effort:
        Opt-in (default False keeps the behaviour every earlier run used).
        When True, Claude gets adaptive thinking plus an explicit
        `output_config.effort = reasoning_effort`. Needed on Claude Opus 5.5,
        where thinking cannot be turned off and effort is the only control
        (its default is medium, so "low" would otherwise be ignored).
    """

    def __init__(self, model: str, provider: str = "auto",
                 reasoning_effort: str = "low",
                 max_output_tokens: int = 16384,
                 use_cache: bool = True,
                 base_url: str | None = None,
                 key_suffix: str | None = None,
                 anthropic_effort: bool = False):
        self.model = model
        self.provider = infer_provider(model) if provider in (None, "auto") else provider
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.use_cache = use_cache
        self.anthropic_effort = anthropic_effort
        # Only used by the `openai_compatible` provider (local vLLM / hosted
        # OpenAI-compatible endpoint). Falls back to OPENAI_COMPAT_BASE_URL.
        self.base_url = base_url
        # Optional key-set selector: with key_suffix="LAB", each provider
        # prefers e.g. ANTHROPIC_API_KEY_LAB and falls back to the plain
        # ANTHROPIC_API_KEY. Lets several people's keys live in one .env.
        self.key_suffix = (key_suffix or os.environ.get("LLM_KEY_SUFFIX") or "").strip()
        # The env var name a key was actually read from (provenance / debugging).
        self.key_source = None

        # Lazily-populated provider SDK handle.
        self._client = None
        self._openai_supports_cache_key = True

        # Token usage from the most recent call, normalised across providers:
        # {"input", "cached_input", "output"}. "cached_input" is the portion of
        # the input billed at the cheaper cache-read rate (a cache hit).
        self.last_usage = None
        # Why the most recent reply ended: Anthropic stop_reason ("end_turn",
        # "max_tokens", "refusal", ...) or OpenAI status / incomplete reason.
        # Lets callers tell a truncated or refused reply from a bad one.
        self.last_stop_reason = None

        self._init_client()

    # -- initialisation ----------------------------------------------------

    def _resolve_key(self, *base_names: str) -> str | None:
        """Return the first non-empty API key among the given env var names,
        preferring the `{NAME}_{key_suffix}` variant when a suffix is set.

        E.g. base_names=("ANTHROPIC_API_KEY",) with key_suffix="LAB" tries
        ANTHROPIC_API_KEY_LAB first, then ANTHROPIC_API_KEY. Records the
        env var it read from on self.key_source."""
        candidates = []
        if self.key_suffix:
            candidates += [f"{n}_{self.key_suffix}" for n in base_names]
        candidates += list(base_names)
        for name in candidates:
            val = os.environ.get(name)
            if val:
                self.key_source = name
                return val
        return None

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            key = self._resolve_key("OPENAI_API_KEY")
            if not key:
                sys.exit(f"No OpenAI key found ({self._key_hint('OPENAI_API_KEY')}). "
                         "Check the root .env file.")
            self._client = OpenAI(api_key=key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            key = self._resolve_key("ANTHROPIC_API_KEY")
            if not key:
                sys.exit(f"No Anthropic key found ({self._key_hint('ANTHROPIC_API_KEY')}). "
                         "Check the root .env file.")
            self._client = Anthropic(api_key=key)
        elif self.provider == "google":
            from google import genai
            # Google AI Studio keys live in GEMINI_API_KEY (fall back to GOOGLE_API_KEY).
            key = self._resolve_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
            if not key:
                sys.exit(f"No Google key found ({self._key_hint('GEMINI_API_KEY')}). "
                         "Check the root .env file.")
            self._client = genai.Client(api_key=key)
        elif self.provider == "openai_compatible":
            # An open-weight model served behind an OpenAI-compatible endpoint —
            # either a local vLLM server (`vllm serve <model> --port 8000`, base
            # url http://localhost:8000/v1) or a hosted provider (Together,
            # Fireworks, DeepInfra, OpenRouter, …). Same SDK, different base_url.
            from openai import OpenAI
            base_url = self.base_url or os.environ.get("OPENAI_COMPAT_BASE_URL")
            if not base_url:
                sys.exit("openai_compatible provider needs a base URL. Set "
                         "OPENAI_COMPAT_BASE_URL (e.g. http://localhost:8000/v1 for "
                         "local vLLM, or a hosted endpoint) or pass --base-url.")
            # vLLM ignores the key; hosted providers read OPENAI_COMPAT_API_KEY.
            key = self._resolve_key("OPENAI_COMPAT_API_KEY") or "EMPTY"
            self._client = OpenAI(api_key=key, base_url=base_url)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _key_hint(self, base_name: str) -> str:
        """Human hint listing the env var names tried, for error messages."""
        if self.key_suffix:
            return f"tried {base_name}_{self.key_suffix} then {base_name}"
        return f"tried {base_name}"

    @staticmethod
    def _key_for(system_prompt: str) -> str:
        return "intro-analysis-" + hashlib.sha256(system_prompt.encode()).hexdigest()[:24]

    def _record_usage(self, input_tokens, cached_input_tokens, output_tokens):
        self.last_usage = {
            "input": input_tokens or 0,
            "cached_input": cached_input_tokens or 0,
            "output": output_tokens or 0,
        }

    @staticmethod
    def _anthropic_usage(u) -> dict:
        """Anthropic splits the prompt into uncached / cache-write / cache-read
        tokens; `input` is their sum, so it means "all prompt tokens" as for
        the other providers. cache_write is billed at 1.25x the input rate."""
        uncached = getattr(u, "input_tokens", 0) or 0
        write = getattr(u, "cache_creation_input_tokens", 0) or 0
        read = getattr(u, "cache_read_input_tokens", 0) or 0
        return {"input": uncached + write + read, "cached_input": read,
                "cache_write": write, "output": getattr(u, "output_tokens", 0) or 0}

    # -- public API --------------------------------------------------------

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw text reply for (system_prompt, user_prompt)."""
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        if self.provider == "google":
            return self._generate_google(system_prompt, user_prompt)
        if self.provider == "openai_compatible":
            return self._generate_openai_compatible(system_prompt, user_prompt)
        raise ValueError(f"Unknown provider: {self.provider}")

    # -- OpenAI ------------------------------------------------------------

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=self.max_output_tokens,
        )
        # OpenAI caches >1024-token prefixes automatically; prompt_cache_key just
        # pins all of a run's calls to the same cache for a better hit rate.
        if self.use_cache and self._openai_supports_cache_key:
            kwargs["prompt_cache_key"] = self._key_for(system_prompt)
        try:
            resp = self._client.responses.create(**kwargs)
        except TypeError:
            # Older SDK without prompt_cache_key — drop it and don't retry it.
            self._openai_supports_cache_key = False
            kwargs.pop("prompt_cache_key", None)
            resp = self._client.responses.create(**kwargs)
        u = getattr(resp, "usage", None)
        if u is not None:
            cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0)
            self._record_usage(getattr(u, "input_tokens", 0), cached,
                               getattr(u, "output_tokens", 0))
        status = getattr(resp, "status", None)
        incomplete = getattr(getattr(resp, "incomplete_details", None), "reason", None)
        self.last_stop_reason = incomplete or status
        return resp.output_text

    # -- Anthropic ---------------------------------------------------------

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        # A cache_control breakpoint on the system block tells Anthropic to cache
        # everything up to and including it (the whole static prompt).
        system_block = {"type": "text", "text": system_prompt}
        if self.use_cache:
            system_block["cache_control"] = {"type": "ephemeral"}

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=[system_block],
            messages=[{"role": "user", "content": user_prompt}],
        )
        self._anthropic_thinking(kwargs)

        # Large max_tokens must stream, or the SDK refuses the request over
        # its HTTP timeout; the final message is identical either way.
        if self.max_output_tokens > 16384:
            with self._client.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
        else:
            resp = self._client.messages.create(**kwargs)
        self.last_stop_reason = getattr(resp, "stop_reason", None)
        u = getattr(resp, "usage", None)
        if u is not None:
            self.last_usage = self._anthropic_usage(u)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

    def _anthropic_thinking(self, params: dict) -> None:
        """Set the thinking / effort fields of an Anthropic request in place."""
        if self.anthropic_effort:
            # Opus 5.5: thinking is always on; depth is set by effort only.
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": self.reasoning_effort}
            return
        # Current Claude models (Opus 4.6+/4.7/4.8, Sonnet 4.6) use ADAPTIVE
        # thinking — the model decides depth. `{"type": "enabled", "budget_tokens"}`
        # is removed on 4.7/4.8 (returns 400), so we never send a fixed budget.
        # low effort → no thinking (omit); medium/high → adaptive thinking on.
        if self.reasoning_effort in ("medium", "high"):
            params["thinking"] = {"type": "adaptive"}

    # -- Google Gemini -----------------------------------------------------

    def _generate_google(self, system_prompt: str, user_prompt: str) -> str:
        # Gemini 2.5 models do prompt caching automatically (implicit caching):
        # because we send the same large system_instruction on every call, the
        # provider detects the repeated prefix and bills it at the cached rate
        # with no extra code. (Caching can't be "turned off" per request; if the
        # user passed --no-cache we simply don't rely on it — behaviour is the
        # same call, the provider just may not get a hit.)
        from google.genai import types

        config_kwargs = dict(
            system_instruction=system_prompt,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        budget = _THINKING_BUDGET.get(self.reasoning_effort, 0)
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)

        try:
            config = types.GenerateContentConfig(**config_kwargs)
            resp = self._client.models.generate_content(
                model=self.model, contents=user_prompt, config=config)
        except Exception:
            # Retry once without thinking_config (some models reject an explicit
            # budget).
            config_kwargs.pop("thinking_config", None)
            config = types.GenerateContentConfig(**config_kwargs)
            resp = self._client.models.generate_content(
                model=self.model, contents=user_prompt, config=config)

        u = getattr(resp, "usage_metadata", None)
        if u is not None:
            self._record_usage(getattr(u, "prompt_token_count", 0),
                               getattr(u, "cached_content_token_count", 0),
                               getattr(u, "candidates_token_count", 0))
        return resp.text

    # -- OpenAI-compatible (open-weight: local vLLM / hosted) ---------------

    def _generate_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        # Open-weight instruct models speak the OpenAI Chat Completions API
        # (not the proprietary Responses API), whether served by vLLM locally or
        # a hosted provider. vLLM applies prefix caching automatically, so the
        # repeated system prompt is reused with no extra parameters.
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_output_tokens,
        )
        try:
            resp = self._client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs)
        except Exception:
            # Some open models / servers don't support response_format — retry
            # plain (the prompt already asks for a JSON object; parse_json copes).
            resp = self._client.chat.completions.create(**kwargs)
        u = getattr(resp, "usage", None)
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            self._record_usage(getattr(u, "prompt_tokens", 0), cached,
                               getattr(u, "completion_tokens", 0))
        return resp.choices[0].message.content

    # -- Batch inference (50% cheaper, async) ------------------------------
    # OpenAI Batch API and Anthropic Message Batches both run requests
    # asynchronously (usually < 1 h, max 24 h) at half the synchronous price —
    # ideal for a non-latency-sensitive generation run of dozens of papers.
    # Only the two hosted providers expose a first-party batch endpoint; the
    # open-weight `openai_compatible` route has no batch discount (self-hosting
    # is already free — run it synchronously with concurrency instead).

    @property
    def supports_batch(self) -> bool:
        return self.provider in ("openai", "anthropic")

    def submit_batch(self, requests: list[dict]) -> str:
        """Submit a batch of generation requests; return the provider batch id.

        Each request is a dict {"custom_id", "system", "user"}."""
        if self.provider == "openai":
            return self._submit_batch_openai(requests)
        if self.provider == "anthropic":
            return self._submit_batch_anthropic(requests)
        raise ValueError(f"Batch inference is not available for provider "
                         f"{self.provider!r} (use openai or anthropic).")

    def poll_batch(self, batch_id: str) -> dict:
        """Return {'status', 'done', counts…} for a submitted batch."""
        if self.provider == "openai":
            b = self._client.batches.retrieve(batch_id)
            rc = getattr(b, "request_counts", None)
            return {
                "status": b.status,
                "done": b.status in ("completed", "failed", "expired", "cancelled"),
                "total": getattr(rc, "total", None),
                "completed": getattr(rc, "completed", None),
                "failed": getattr(rc, "failed", None),
            }
        if self.provider == "anthropic":
            b = self._client.messages.batches.retrieve(batch_id)
            rc = b.request_counts
            return {
                "status": b.processing_status,
                "done": b.processing_status == "ended",
                "succeeded": getattr(rc, "succeeded", None),
                "errored": getattr(rc, "errored", None),
                "processing": getattr(rc, "processing", None),
            }
        raise ValueError(f"Batch inference is not available for provider {self.provider!r}.")

    def fetch_batch(self, batch_id: str) -> dict:
        """Return {custom_id: {'text'|'error', 'usage'}} for a finished batch."""
        if self.provider == "openai":
            return self._fetch_batch_openai(batch_id)
        if self.provider == "anthropic":
            return self._fetch_batch_anthropic(batch_id)
        raise ValueError(f"Batch inference is not available for provider {self.provider!r}.")

    # -- OpenAI batch ------------------------------------------------------

    def _submit_batch_openai(self, requests: list[dict]) -> str:
        lines = []
        for r in requests:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["user"]},
                ],
                "max_completion_tokens": self.max_output_tokens,
            }
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
            if self.use_cache:
                body["prompt_cache_key"] = self._key_for(r["system"])
            lines.append(json.dumps({
                "custom_id": r["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))
        content = ("\n".join(lines) + "\n").encode("utf-8")
        upload = self._client.files.create(
            file=("intro_generation_batch.jsonl", content), purpose="batch")
        batch = self._client.batches.create(
            input_file_id=upload.id, endpoint="/v1/chat/completions",
            completion_window="24h")
        return batch.id

    def _fetch_batch_openai(self, batch_id: str) -> dict:
        b = self._client.batches.retrieve(batch_id)
        if not getattr(b, "output_file_id", None):
            raise RuntimeError(f"OpenAI batch {batch_id} status={b.status} has no "
                               f"output yet.")
        text = self._client.files.content(b.output_file_id).text
        out: dict = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            resp = obj.get("response")
            if resp and resp.get("status_code") == 200:
                body = resp["body"]
                msg = body["choices"][0]["message"]["content"]
                out[cid] = {"text": msg, "usage": body.get("usage")}
            else:
                out[cid] = {"error": obj.get("error") or (resp or {}).get("status_code")}
        return out

    # -- Anthropic batch ---------------------------------------------------

    def _submit_batch_anthropic(self, requests: list[dict]) -> str:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        reqs = []
        for r in requests:
            system_block = {"type": "text", "text": r["system"]}
            if self.use_cache:
                system_block["cache_control"] = {"type": "ephemeral"}
            params: dict = dict(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=[system_block],
                messages=[{"role": "user", "content": r["user"]}],
            )
            self._anthropic_thinking(params)
            reqs.append(Request(
                custom_id=r["custom_id"],
                params=MessageCreateParamsNonStreaming(**params)))
        batch = self._client.messages.batches.create(requests=reqs)
        return batch.id

    def _fetch_batch_anthropic(self, batch_id: str) -> dict:
        out: dict = {}
        for result in self._client.messages.batches.results(batch_id):
            cid = result.custom_id
            r = result.result
            if r.type == "succeeded":
                msg = r.message
                text = "".join(
                    block.text for block in msg.content
                    if getattr(block, "type", None) == "text")
                u = getattr(msg, "usage", None)
                out[cid] = {"text": text, "usage": self._anthropic_usage(u) if u else None}
            else:
                out[cid] = {"error": r.type}
        return out
