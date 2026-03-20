"""
llm.py — LLM Setup and Generation Module
==========================================
Manages LLM connections and answer generation.

Supports three providers:
  "openai" : GPT-4o-mini via OpenAI API (paid, highest quality)
  "groq"   : llama3-70b-8192 via Groq API (FREE, very fast)
  "local"  : ollama models (completely offline, no API needed)

Provider is selected via LLM_PROVIDER in .env — no code changes needed.

The LLM is the FINAL step in the pipeline:
  retrieved chunks → reranked → citation prompt → LLM → cited answer
"""

import time  # Timing generation
from typing import Optional, Generator  # Type hints

from loguru import logger

from src.config import (
    LLM_PROVIDER,  # "openai" or "groq" or "local"
    LLM_TEMPERATURE,  # 0.1 — factual, consistent
    LLM_MAX_TOKENS,  # 1024 — enough for a well-cited answer
    OPENAI_API_KEY,  # needed if provider = "openai"
    GROQ_API_KEY,  # needed if provider = "groq"
    OPENAI_LLM_MODEL,  # "gpt-4o-mini"
    GROQ_LLM_MODEL,  # "llama3-70b-8192"
    CONTEXT_WINDOW_LIMITS,  # max tokens per model
)


# ══════════════════════════════════════════════════════════════════
# LLM FACTORY — returns correct LLM based on config
# ══════════════════════════════════════════════════════════════════


def get_llm(provider: Optional[str] = None):
    """
    Factory function — returns configured LLM instance.

    Reads LLM_PROVIDER from config (set in .env) and returns
    the appropriate LangChain LLM object. All returned objects
    have the same interface: .invoke(), .stream()

    Args:
        provider: override config value ("openai", "groq", "local")
                  If None, uses LLM_PROVIDER from .env

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

    Model: gpt-4o-mini
      - Context: 128K tokens
      - Cost: $0.15 per million input tokens
      - Quality: Highest available for this price point
      - Streaming: supported

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
                "Get your key: https://platform.openai.com/api-keys\n"
                "Or use free Groq: set LLM_PROVIDER=groq in .env"
            )

        llm = ChatOpenAI(
            model=OPENAI_LLM_MODEL,  # "gpt-4o-mini"
            temperature=LLM_TEMPERATURE,  # 0.1
            max_tokens=LLM_MAX_TOKENS,  # 1024
            openai_api_key=OPENAI_API_KEY,
            streaming=True,  # enable token streaming
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

    Model: llama3-70b-8192
      - Context: 8,192 tokens
      - Cost: FREE (as of 2025)
      - Speed: ~800 tokens/second (10x faster than OpenAI)
      - Quality: Very good for RAG tasks

    WHY Groq for development:
      - Zero cost during experimentation
      - Fastest inference available
      - Good enough quality for building and testing

    Returns:
        ChatGroq instance

    Raises:
        ValueError  : API key missing
        RuntimeError: connection fails
    """
    try:
        # Use direct Groq client instead of LangChain wrapper
        # to avoid torch.classes conflict on Streamlit Cloud
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is missing in .env\n"
                "Get your FREE key: https://console.groq.com"
            )

        # Set environment variable explicitly before init
        import os
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY

        llm = ChatGroq(
            model        = GROQ_LLM_MODEL,
            temperature  = LLM_TEMPERATURE,
            max_tokens   = LLM_MAX_TOKENS,
            groq_api_key = GROQ_API_KEY,
            # Disable streaming on cloud to avoid timeout
            streaming    = False,
        )

        logger.info(
            f"Groq LLM ready | Model: {GROQ_LLM_MODEL} | "
            f"Temperature: {LLM_TEMPERATURE}"
        )
        return llm

    except (ValueError, ImportError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq LLM: {e}") from e

def _get_local_llm():
    """
    Creates a local Ollama LLM instance (fully offline).

    Requires Ollama installed and running:
      1. Install: https://ollama.ai
      2. Pull model: ollama pull mistral
      3. Start server: ollama serve  (runs on localhost:11434)

    No API key needed. Completely private — data never leaves
    your machine.

    Returns:
        ChatOllama instance

    Raises:
        RuntimeError: Ollama not running or model not pulled
    """
    try:
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(
            model="mistral",  # or "llama3", "phi3", etc.
            temperature=LLM_TEMPERATURE,
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
            f"Make sure Ollama is installed and running:\n"
            f"  1. Install: https://ollama.ai\n"
            f"  2. Pull model: ollama pull mistral\n"
            f"  3. Start: ollama serve"
        ) from e


# ══════════════════════════════════════════════════════════════════
# ANSWER GENERATION
# ══════════════════════════════════════════════════════════════════


def generate_answer(
    llm,
    prompt: str,
    stream: bool = False,
) -> str:
    """
    Sends a prompt to the LLM and returns the complete response.

    Handles both streaming and non-streaming modes.
    In non-streaming mode: waits for full response, returns string.
    In streaming mode: collects all chunks, returns combined string.

    Args:
        llm   : LLM instance from get_llm()
        prompt: complete prompt string (from citation_enforcer)
        stream: if True, prints tokens as they arrive (for terminal)

    Returns:
        Complete response string from the LLM

    Raises:
        ValueError  : empty prompt
        RuntimeError: API call fails (rate limit, network, etc.)
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
            # ── Streaming mode ──────────────────────────────────────
            # .stream() yields AIMessageChunk objects one by one
            # Each chunk has a .content attribute with the token text
            full_response = ""
            for chunk in llm.stream(prompt):
                token = chunk.content
                full_response += token
                # Print without newline so tokens appear inline
                print(token, end="", flush=True)
            print()  # newline after streaming complete

        else:
            # ── Non-streaming mode ──────────────────────────────────
            # .invoke() sends prompt, waits, returns AIMessage
            response = llm.invoke(prompt)
            full_response = response.content

        elapsed = time.time() - start_time

        # Count tokens in response (approximate)
        response_tokens = len(full_response.split()) * 1.3  # rough estimate

        logger.info(
            f"Answer generated in {elapsed:.2f}s | "
            f"Response: {len(full_response)} chars | "
            f"~{response_tokens:.0f} tokens"
        )

        return full_response

    except ValueError:
        raise
    except Exception as e:
        # Provide helpful error messages for common failures
        error_str = str(e).lower()

        if "rate limit" in error_str:
            raise RuntimeError(
                f"Rate limit exceeded.\n"
                f"Wait 60 seconds and try again, or switch providers.\n"
                f"Original error: {e}"
            ) from e

        elif "api key" in error_str or "authentication" in error_str:
            raise RuntimeError(
                f"API key invalid or expired.\n"
                f"Check your .env file and verify the key is correct.\n"
                f"Original error: {e}"
            ) from e

        elif "context" in error_str or "token" in error_str:
            raise RuntimeError(
                f"Context window exceeded.\n"
                f"The prompt is too long for this model.\n"
                f"Try reducing RERANK_TOP_K in .env to pass fewer chunks.\n"
                f"Original error: {e}"
            ) from e

        else:
            raise RuntimeError(f"LLM generation failed: {e}") from e


def generate_answer_stream(
    llm,
    prompt: str,
) -> Generator[str, None, None]:
    """
    Generator function that yields tokens one by one.
    Used by Streamlit's st.write_stream() for live typing effect.

    This is the streaming version used in the UI.
    Each yield returns one token (a few characters).

    Args:
        llm   : LLM instance from get_llm()
        prompt: complete prompt string

    Yields:
        Individual token strings as they are generated

    Raises:
        RuntimeError: API call fails

    Usage in Streamlit:
        response = st.write_stream(generate_answer_stream(llm, prompt))
    """
    try:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        logger.info(f"Starting streaming generation | " f"Prompt: {len(prompt)} chars")

        # .stream() yields AIMessageChunk objects
        # Each chunk.content is a small string (1-5 tokens)
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        # Yield error message as part of stream
        # This way Streamlit shows the error inline instead of crashing
        yield f"\n\n[Error generating response: {str(e)}]"


# ══════════════════════════════════════════════════════════════════
# TOKEN COUNTING UTILITIES
# ══════════════════════════════════════════════════════════════════


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Counts the exact number of tokens in a text string.

    Uses tiktoken — OpenAI's tokenizer library.
    Same tokenizer the models use internally.
    Accurate for OpenAI models, approximate for others.

    WHY count tokens?
    - Ensure prompt fits in model's context window
    - Estimate API cost before sending
    - Debug "context too long" errors

    Args:
        text : string to count tokens for
        model: model name for correct tokenizer selection

    Returns:
        Integer token count

    Example:
        count_tokens("Hello world") → 2
        count_tokens("The revenue was $4.2 million") → 7
    """
    try:
        import tiktoken

        # Get the correct encoding for this model
        # cl100k_base: used by GPT-3.5, GPT-4, GPT-4o
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for non-OpenAI models (Groq, local)
            encoding = tiktoken.get_encoding("cl100k_base")

        token_count = len(encoding.encode(text))
        return token_count

    except ImportError:
        # tiktoken not installed — use word count approximation
        # 1 token ≈ 0.75 words (rough but usable)
        approx = int(len(text.split()) / 0.75)
        logger.warning(
            f"tiktoken not installed — using approximate token count: {approx}\n"
            f"Fix: pip install tiktoken"
        )
        return approx
    except Exception as e:
        logger.warning(f"Token counting failed: {e}. Using approximation.")
        return int(len(text.split()) / 0.75)


def check_context_fits(
    prompt: str,
    model: str = GROQ_LLM_MODEL,
    safety_margin: int = 500,
) -> bool:
    """
    Checks if a prompt fits within a model's context window.

    Leaves a safety_margin of 500 tokens for the response.
    If prompt + 500 > context_limit → returns False (won't fit).

    Args:
        prompt       : the complete prompt string
        model        : model name to check limit for
        safety_margin: tokens to reserve for the response

    Returns:
        True  : prompt fits (safe to send)
        False : prompt too long (need to reduce chunks)

    Usage:
        if not check_context_fits(prompt, GROQ_LLM_MODEL):
            # reduce number of chunks passed to LLM
            chunks = chunks[:3]   # trim from 5 to 3
    """
    # Get this model's context window limit
    context_limit = CONTEXT_WINDOW_LIMITS.get(model, 8192)

    # Count tokens in the prompt
    prompt_tokens = count_tokens(prompt, model)

    # Check: prompt + reserved response space must fit
    fits = (prompt_tokens + safety_margin) <= context_limit

    logger.info(
        f"Context check | "
        f"Model: {model} | "
        f"Prompt: {prompt_tokens} tokens | "
        f"Limit: {context_limit} | "
        f"Fits: {fits}"
    )

    if not fits:
        logger.warning(
            f"Prompt too long: {prompt_tokens} tokens + "
            f"{safety_margin} safety margin > {context_limit} limit.\n"
            f"Reduce RERANK_TOP_K in .env to pass fewer chunks to the LLM."
        )

    return fits


# ══════════════════════════════════════════════════════════════════
# QUICK TEST FUNCTION
# ══════════════════════════════════════════════════════════════════


def test_llm_connection(provider: Optional[str] = None) -> bool:
    """
    Tests LLM connectivity with a simple ping message.
    Used at app startup to verify the API key works.

    Args:
        provider: which provider to test (default: from config)

    Returns:
        True  : LLM responded successfully
        False : connection failed (logged with reason)
    """
    try:
        llm = get_llm(provider)
        response = generate_answer(llm, "Say 'OK' and nothing else.")

        if response and len(response.strip()) > 0:
            logger.info(
                f"LLM connection test passed | " f"Response: '{response.strip()[:50]}'"
            )
            return True
        else:
            logger.warning("LLM returned empty response during test.")
            return False

    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False
