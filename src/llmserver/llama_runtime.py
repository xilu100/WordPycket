from __future__ import annotations

import json
import sys
from typing import Any


def is_llama_cleanup_error(unraisable: Any) -> bool:
    return (
        unraisable.exc_type is AttributeError
        and "'LlamaModel' object has no attribute 'sampler'"
        in str(unraisable.exc_value)
    )


def install_llama_cleanup_error_filter() -> None:
    current_hook = sys.unraisablehook
    if getattr(current_hook, "_wordpycket_llama_filter", False):
        return

    def filter_llama_cleanup_error(unraisable: Any) -> None:
        if not is_llama_cleanup_error(unraisable):
            current_hook(unraisable)

    filter_llama_cleanup_error._wordpycket_llama_filter = True  # type: ignore[attr-defined]
    sys.unraisablehook = filter_llama_cleanup_error


def call_model(
    llm: Any,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 200,
) -> str:
    messages = [
        {
            "role": "system",
            "content": system_prompt
            or "You generate concise English vocabulary examples. Return only valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.4,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return extract_chat_content(response)
    except (KeyError, TypeError, ValueError, AttributeError):
        response = llm.create_completion(
            prompt=build_completion_prompt(prompt, system_prompt),
            temperature=0.4,
            max_tokens=max_tokens,
            stop=["\n\n"],
        )
        return extract_completion_text(response)


def call_smoke_model(llm: Any) -> str:
    messages = [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": 'Return exactly {"ok": true}.'},
    ]
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=16,
            response_format={"type": "json_object"},
        )
        return extract_chat_content(response)
    except (KeyError, TypeError, ValueError, AttributeError):
        response = llm.create_completion(
            prompt='Return only valid JSON.\n\nReturn exactly {"ok": true}.\n\nJSON:',
            temperature=0,
            max_tokens=16,
            stop=["\n\n"],
        )
        return extract_completion_text(response)


def extract_chat_content(response: Any) -> str:
    choices = response["choices"]
    message = choices[0]["message"]
    content = message["content"]
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )
    return str(content)


def extract_completion_text(response: Any) -> str:
    return str(response["choices"][0]["text"])


def build_completion_prompt(prompt: str, system_prompt: str | None = None) -> str:
    return (
        f"{system_prompt or 'You generate concise English vocabulary examples. Return only valid JSON.'}\n\n"
        f"{prompt}\n\nJSON:"
    )
