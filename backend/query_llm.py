import os
import logging
from typing import Any
from typing import Dict
from typing import Generator
from typing import List

import gradio as gr
import openai
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Safety checks for API keys
if not OPENAI_KEY or OPENAI_KEY == "your_openai_key_here_but_use_hf_secrets_instead":
    logger.warning("OpenAI API key not properly configured")
    OPENAI_KEY = None

if not HF_TOKEN or HF_TOKEN == "your_hf_token_here_but_use_hf_secrets_instead":
    logger.warning("HuggingFace token not properly configured")

TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("HF_MODEL"))

HF_CLIENT = InferenceClient(os.getenv("HF_MODEL"), token=HF_TOKEN)
OAI_CLIENT = openai.Client(api_key=OPENAI_KEY) if OPENAI_KEY else None

HF_GENERATE_KWARGS = {
    "temperature": max(float(os.getenv("TEMPERATURE", 0.9)), 1e-2),
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", 256)),
    "top_p": float(os.getenv("TOP_P", 0.6)),
    "repetition_penalty": float(os.getenv("REP_PENALTY", 1.2)),
    "do_sample": bool(os.getenv("DO_SAMPLE", True)),
}

OAI_GENERATE_KWARGS = {
    "temperature": max(float(os.getenv("TEMPERATURE", 0.9)), 1e-2),
    "max_tokens": int(os.getenv("MAX_NEW_TOKENS", 256)),
    "top_p": float(os.getenv("TOP_P", 0.6)),
    "frequency_penalty": max(-2, min(float(os.getenv("FREQ_PENALTY", 0)), 2)),
}


def format_prompt(message: str, api_kind: str):
    """
    Formats the given message using a chat template.

    Args:
        message (str): The user message to be formatted.
        api_kind (str): LLM API provider.
    Returns:
        str: Formatted message after applying the chat template.
    """

    messages: List[Dict[str, Any]] = [{"role": "user", "content": message}]

    if api_kind == "openai":
        return messages
    elif api_kind == "hf":
        return TOKENIZER.apply_chat_template(messages, tokenize=False)
    else:
        raise ValueError("API is not supported")


def generate_hf(prompt: str) -> Generator[str, None, str]:
    """
    Generate a sequence of tokens based on a given prompt using HuggingFace client.

    Args:
        prompt (str): The prompt for the text generation.
    Returns:
        Generator[str, None, str]: A generator yielding chunks of generated text.
    """

    formatted_prompt = format_prompt(prompt, "hf")
    formatted_prompt = formatted_prompt.encode("utf-8").decode("utf-8")

    try:
        stream = HF_CLIENT.text_generation(
            formatted_prompt,
            **HF_GENERATE_KWARGS,
            stream=True,
            details=True,
            return_full_text=False,
        )
        output = ""
        for response in stream:
            output += response.token.text
            yield output

    except Exception as e:
        if "Too Many Requests" in str(e):
            raise gr.Error(f"HuggingFace API rate limit reached. Please try again later.")
        elif "Authorization header is invalid" in str(e):
            raise gr.Error(
                "HuggingFace authentication error. Please check the configuration."
            )
        else:
            logger.error(f"HuggingFace API error: {str(e)}")
            raise gr.Error(f"HuggingFace API error: {str(e)}")


def generate_openai(prompt: str) -> Generator[str, None, str]:
    """
    Generate a sequence of tokens based on a given prompt using OpenAI client.

    Args:
        prompt (str): The initial prompt for the text generation.
    Returns:
        Generator[str, None, str]: A generator yielding chunks of generated text.
    """
    
    if not OAI_CLIENT:
        raise gr.Error("OpenAI API is not configured. Please use HuggingFace instead.")
    
    formatted_prompt = format_prompt(prompt, "openai")

    try:
        # Log the request for monitoring
        logger.info(f"OpenAI request initiated - tokens: ~{len(prompt)//4}")
        
        stream = OAI_CLIENT.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=formatted_prompt,
            **OAI_GENERATE_KWARGS,
            stream=True,
        )
        output = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                output += chunk.choices[0].delta.content
                yield output

    except Exception as e:
        if "rate limit" in str(e).lower():
            raise gr.Error("OpenAI rate limit reached. Please try again later or use HuggingFace.")
        elif "insufficient" in str(e).lower() or "quota" in str(e).lower():
            raise gr.Error("OpenAI quota exceeded. Please use HuggingFace instead.")
        elif "invalid" in str(e).lower() and "key" in str(e).lower():
            raise gr.Error("OpenAI authentication error. Please use HuggingFace instead.")
        else:
            logger.error(f"OpenAI API error: {str(e)}")
            raise gr.Error("OpenAI API error. Please try HuggingFace instead.")
