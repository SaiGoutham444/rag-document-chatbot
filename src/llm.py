"""
llm.py — LLM Setup and Generation Module
==========================================
Manages LLM connections and answer generation.

Supports three providers:
  "openai" : GPT-4o-mini via OpenAI API (paid, highest quality)
  "groq"   : llama-3.3-70b via Groq API (FREE, very fast)
  "local"  : ollama models (completely offline, no API needed)
"""

import os
import time
from typing import Optional, Generator

from loguru import logger

from src.config import (
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    OPENAI_LLM_MODEL,
    GROQ_LLM_MODEL,
    CONTEXT_WINDOW_LIMITS,
)


# ══════════════════════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════════════════════

def get_llm(provider: Optional[str] = None):
    """
    Factory function — returns configured LLM instance.

    Args:
        provider: override config value ("openai", "groq", "local")

    Returns:
        LangChain LLM object with .invoke() and .stream() methods

    Raises:
        ValueError  : unknown provider or missing API key
        RuntimeError: LLM fails to initialize
    """
    selected = provider or LLM_PROVIDER
    logger.info(f"Initializing LLM | Provider: '{selected}'")

    if selected == "openai":
        return _get_openai_llm()
    elif selected == "groq":
        return _get_groq_llm()
    elif selected == "local":
        return _get_local_llm()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{selected}'\n"
            f"Valid options: 'openai', 'groq', 'local'\n"
            f"Check LLM_PROVIDER in your .env file."
        )


def _get_openai_llm():
    """
    Creates OpenAI GPT-4o-mini LLM instance.

    Returns:
        ChatOpenAI instance

    Raises:
        ValueError  : API key missing
        RuntimeError: connection fails
    """
    try:
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is missing in .env\n"
                "Get your key: https://platform.openai.com/api-keys"
            )

        llm = ChatOpenAI(
            model          = OPENAI_LLM_MODEL,
            temperature    = LLM_TEMPERATURE,
            max_tokens     = LLM_MAX_TOKENS,
            openai_api_key = OPENAI_API_KEY,
            streaming      = True,
        )

        logger.info(
            f"OpenAI LLM ready | "
            f"Model: {OPENAI_LLM_MODEL} | "
            f"Temperature: {LLM_TEMPERATURE}"
        )
        return llm

    except (ValueError, ImportError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI LLM: {e}") from e


def _get_groq_llm():
    """
    Creates Groq LLM instance (FREE, ultra-fast inference).

    Uses monkey-patching to fix the 'proxies' error that occurs
    on Streamlit Cloud with newer httpx versions.

    groq==0.9.0 + langchain-groq==0.1.9 pass 'proxies' to httpx
    internally. Newer httpx removed this parameter → crash.
    We patch httpx.Client.__init__ and httpx.AsyncClient.__init__
    to silently drop 'proxies' before it causes an error.

    Returns:
        ChatGroq instance

    Raises:
        ValueError  : API key missing
        RuntimeError: connection fails
    """
    try:
        # ── STEP 1: Monkey-patch httpx BEFORE importing langchain_groq
        # This must happen before any groq/httpx initialization
        import httpx

        # Patch synchronous Client
        _orig_client = httpx.Client.__init__

        def _patched_client(self, *args, **kwargs):
            kwargs.pop("proxies", None)
            _orig_client(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_client

        # Patch asynchronous AsyncClient
        _orig_async = httpx.AsyncClient.__init__

        def _patched_async(self, *args, **kwargs):
            kwargs.pop("proxies", None)
            _orig_async(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = _patched_async

        # ── STEP 2: Now safe to import and use langchain_groq
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is missing in .env\n"
                "Get your FREE key: https://console.groq.com"
            )

        # Set in os.environ so groq SDK can find it
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY

        # ── STEP 3: Initialize the LLM
        llm = ChatGroq(
            model        = GROQ_LLM_MODEL,
            temperature  = LLM_TEMPERATURE,
            max_tokens   = LLM_MAX_TOKENS,
            groq_api_key = GROQ_API_KEY,
        )

        logger.info(
            f"Groq LLM ready | "
            f"Model: {GROQ_LLM_MODEL} | "
            f"Temperature: {LLM_TEMPERATURE} | "
            f"Context: {CONTEXT_WINDOW_LIMITS.get(GROQ_LLM_MODEL, 128000)} tokens"
        )
        return llm

    except (ValueError, ImportError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Groq LLM: {e}\n"
            f"Check your GROQ_API_KEY and internet connection."
        ) from e


def _get_local_llm():
    """
    Creates a local Ollama LLM instance (fully offline).

    Requires:
      1. Install Ollama: https://ollama.ai
      2. Pull model: ollama pull mistral
      3. Start server: ollama serve

    Returns:
        ChatOllama instance

    Raises:
        RuntimeError: Ollama not running or model not pulled
    """
    try:
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(
            model       = "mistral",
            temperature = LLM_TEMPERATURE,
        )

        logger.info(
            "Local Ollama LLM ready | "
            "Model: mistral | "
            "Make sure 'ollama serve' is running"
        )
        return llm

    except ImportError:
        raise RuntimeError(
            "langchain-community not installed.\n"
            "Fix: pip install langchain-community"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to local Ollama: {e}\n"
            f"Make sure Ollama is installed and running."
        ) from e


# ══════════════════════════════════════════════════════════════════
# ANSWER GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_answer(
    llm,
    prompt : str,
    stream : bool = False,
) -> str:
    """
    Sends a prompt to the LLM and returns the complete response.

    Args:
        llm   : LLM instance from get_llm()
        prompt: complete prompt string
        stream: if True, prints tokens as they arrive

    Returns:
        Complete response string from the LLM

    Raises:
        ValueError  : empty prompt
        RuntimeError: API call fails
    """
    try:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        logger.info(
            f"Generating answer | "
            f"Prompt: {len(prompt)} chars | "
            f"Stream: {stream}"
        )
        start_time = time.time()

        if stream:
            full_response = ""
            for chunk in llm.stream(prompt):
                token = chunk.content
                full_response += token
                print(token, end="", flush=True)
            print()
        else:
            response      = llm.invoke(prompt)
            full_response = response.content

        elapsed         = time.time() - start_time
        response_tokens = len(full_response.split()) * 1.3

        logger.info(
            f"Answer generated in {elapsed:.2f}s | "
            f"Response: {len(full_response)} chars | "
            f"~{response_tokens:.0f} tokens"
        )
        return full_response

    except ValueError:
        raise
    except Exception as e:
        error_str = str(e).lower()

        if "rate limit" in error_str:
            raise RuntimeError(
                f"Rate limit exceeded. Wait 60 seconds and try again.\n"
                f"Original error: {e}"
            ) from e
        elif "api key" in error_str or "authentication" in error_str:
            raise RuntimeError(
                f"API key invalid or expired. Check your .env file.\n"
                f"Original error: {e}"
            ) from e
        elif "context" in error_str or "token" in error_str:
            raise RuntimeError(
                f"Context window exceeded. Reduce RERANK_TOP_K in .env.\n"
                f"Original error: {e}"
            ) from e
        else:
            raise RuntimeError(f"LLM generation failed: {e}") from e


def generate_answer_stream(
    llm,
    prompt: str,
) -> Generator[str, None, None]:
    """
    Generator that yields tokens one by one for Streamlit streaming.

    Usage in Streamlit:
        response = st.write_stream(generate_answer_stream(llm, prompt))

    Args:
        llm   : LLM instance from get_llm()
        prompt: complete prompt string

    Yields:
        Individual token strings as generated
    """
    try:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        logger.info(
            f"Starting streaming generation | "
            f"Prompt: {len(prompt)} chars"
        )

        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        yield f"\n\n[Error generating response: {str(e)}]"


# ══════════════════════════════════════════════════════════════════
# TOKEN COUNTING UTILITIES
# ══════════════════════════════════════════════════════════════════

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Counts tokens in a text string using tiktoken.

    Args:
        text : string to count tokens for
        model: model name for correct tokenizer

    Returns:
        Integer token count
    """
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    except ImportError:
        approx = int(len(text.split()) / 0.75)
        logger.warning(f"tiktoken not installed — approximating: {approx}")
        return approx
    except Exception as e:
        logger.warning(f"Token counting failed: {e}. Using approximation.")
        return int(len(text.split()) / 0.75)


def check_context_fits(
    prompt        : str,
    model         : str = GROQ_LLM_MODEL,
    safety_margin : int = 500,
) -> bool:
    """
    Checks if a prompt fits within a model's context window.

    Args:
        prompt       : the complete prompt string
        model        : model name to check limit for
        safety_margin: tokens to reserve for the response

    Returns:
        True  : prompt fits
        False : prompt too long
    """
    context_limit = CONTEXT_WINDOW_LIMITS.get(model, 8192)
    prompt_tokens = count_tokens(prompt, model)
    fits          = (prompt_tokens + safety_margin) <= context_limit

    logger.info(
        f"Context check | "
        f"Model: {model} | "
        f"Prompt: {prompt_tokens} tokens | "
        f"Limit: {context_limit} | "
        f"Fits: {fits}"
    )

    if not fits:
        logger.warning(
            f"Prompt too long: {prompt_tokens} + {safety_margin} "
            f"> {context_limit}. Reduce RERANK_TOP_K in .env."
        )

    return fits


def test_llm_connection(provider: Optional[str] = None) -> bool:
    """
    Tests LLM connectivity with a simple ping.

    Args:
        provider: which provider to test

    Returns:
        True if connected, False if failed
    """
    try:
        llm      = get_llm(provider)
        response = generate_answer(llm, "Say 'OK' and nothing else.")

        if response and len(response.strip()) > 0:
            logger.info(
                f"LLM connection test passed | "
                f"Response: '{response.strip()[:50]}'"
            )
            return True
        else:
            logger.warning("LLM returned empty response during test.")
            return False

    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False