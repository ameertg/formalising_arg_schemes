"""
Logging configuration using loguru.
Provides structured logging for the LLMProver pipeline.
"""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure log format
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Simpler format for file (without ANSI colors)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# Add console handler (INFO level)
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level="INFO",
    colorize=True,
)

# Add file handler for all logs (DEBUG level)
logger.add(
    LOGS_DIR / "llmprover_{time:YYYY-MM-DD}.log",
    format=FILE_FORMAT,
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    encoding="utf-8",
)

# Add separate file for Isabelle server communication only (DEBUG level)
logger.add(
    LOGS_DIR / "isabelle_server_{time:YYYY-MM-DD}.log",
    format=FILE_FORMAT,
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    encoding="utf-8",
    filter=lambda record: record["extra"].get("isabelle_server", False),
)

# Add separate file for LLM prompts and raw outputs (DEBUG level)
logger.add(
    LOGS_DIR / "llm_prompts_{time:YYYY-MM-DD}.log",
    format=FILE_FORMAT,
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    encoding="utf-8",
    filter=lambda record: record["extra"].get("llm_prompt_log", False),
)


def get_logger(name: str):
    """Get a logger with the given name.

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        A loguru logger instance bound to the given name
    """
    return logger.bind(name=name)


def log_isabelle_interaction(direction: str, content: str, context: dict = None):
    """Log Isabelle-LLM interaction with special formatting.

    Args:
        direction: 'isabelle_to_llm' or 'llm_to_isabelle'
        content: The content being communicated
        context: Optional context dict (theory_name, iteration, etc.)
    """
    ctx_str = ""
    if context:
        ctx_str = " | ".join(f"{k}={v}" for k, v in context.items())

    logger.bind(isabelle_interaction=True).info(
        f"[{direction.upper()}] {ctx_str}\n{'-'*60}\n{content}\n{'-'*60}"
    )


def log_isabelle_response(response_type: str, response_body: str, context: dict = None):
    """Log Isabelle server response.

    Args:
        response_type: The type of response (OK, FINISHED, FAILED, etc.)
        response_body: The response body content
        context: Optional context dict
    """
    ctx_str = ""
    if context:
        ctx_str = " | ".join(f"{k}={v}" for k, v in context.items())

    level = "ERROR" if response_type == "FAILED" else "DEBUG"
    logger.bind(isabelle_interaction=True).log(
        level,
        f"[ISABELLE_RESPONSE] type={response_type} | {ctx_str}\n{response_body[:1000] if len(response_body) > 1000 else response_body}"
    )


def log_llm_prompt(prompt_name: str, system_prompt: str, user_prompt: str, context: dict = None):
    """Log LLM prompt being sent.

    Args:
        prompt_name: Name of the prompt template
        system_prompt: The system prompt content
        user_prompt: The user prompt content
        context: Optional context dict
    """
    ctx_str = ""
    if context:
        ctx_str = " | ".join(f"{k}={v}" for k, v in context.items())

    logger.bind(llm_prompt_log=True).info(
        f"[LLM_PROMPT] {prompt_name} | {ctx_str}\n"
        f"{'='*60}\n"
        f"SYSTEM PROMPT:\n{system_prompt}\n"
        f"{'-'*60}\n"
        f"USER PROMPT:\n{user_prompt}\n"
        f"{'='*60}"
    )


def log_llm_response(prompt_name: str, response: str, context: dict = None):
    """Log LLM response received.

    Args:
        prompt_name: Name of the prompt that generated this response
        response: The LLM's response
        context: Optional context dict
    """
    ctx_str = ""
    if context:
        ctx_str = " | ".join(f"{k}={v}" for k, v in context.items())

    logger.bind(llm_prompt_log=True).info(
        f"[LLM_RESPONSE] {prompt_name} | {ctx_str}\n"
        f"{'='*60}\n"
        f"{response}\n"
        f"{'='*60}"
    )
