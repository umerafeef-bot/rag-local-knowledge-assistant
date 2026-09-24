"""Context assembly and the prompt template.

Hand-written, not delegated to a framework, for the same reason as the chunker:
this is the part that decides whether an answer is grounded, so it should be
readable and testable. Two functions, one job each.

``build_context`` packs retrieved chunks into a numbered, source-labelled block
while respecting a character budget. ``build_prompt`` wraps that block and the
question into the text sent to the local model.

Budgeting matters: the tokenizer will happily accept a prompt larger than the
model's context window, and Ollama then truncates it *silently* -- which
typically throws away the question, the end of the context, or both, leaving a
confidently wrong answer. So the context is capped in characters well below the
configured window, and anything dropped is reported back to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.services.retrieval import RetrievedChunk

#: The sentence the model is told to reply with when the context is not enough.
#: The application reuses it verbatim when retrieval itself finds nothing, so a
#: refusal reads the same however it was reached.
INSUFFICIENT_CONTEXT_ANSWER = "I could not find that in the provided documents."

SYSTEM_PROMPT = (
    "You are DocuAsk, an assistant that answers questions about the user's own documents.\n"
    "Follow these rules exactly:\n"
    "1. Answer ONLY from the numbered context in the user message. Never use outside "
    "knowledge and never guess.\n"
    '2. If the context does not contain the answer, reply with exactly this sentence: "'
    f"{INSUFFICIENT_CONTEXT_ANSWER}"
    '"\n'
    "3. Cite the context blocks you used with their numbers, for example [1] or [2][3].\n"
    "4. Keep the answer to at most three sentences."
)

CONTEXT_HEADER = "CONTEXT FROM THE USER'S DOCUMENTS"
QUESTION_HEADER = "QUESTION"
BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class BuiltContext:
    """The numbered context block and what it cost."""

    text: str
    used: tuple[RetrievedChunk, ...]
    chars: int
    #: True when the budget forced something to be left out (or a single
    #: oversized chunk to be cut). Surfaced in the API response so a shortened
    #: context is visible rather than guessed at.
    truncated: bool


def format_block(number: int, chunk: RetrievedChunk) -> str:
    """One numbered context block, labelled with its origin.

    The label gives the model the file and page so its prose can name them, and
    gives a human reading the prompt a way to check the citation.
    """
    return f"[{number}] (file: {chunk.filename}, page {chunk.page})\n{chunk.text.strip()}"


def build_context(hits: Sequence[RetrievedChunk], *, max_chars: int) -> BuiltContext:
    """Pack the best hits into at most ``max_chars`` characters.

    Hits are kept in rank order -- the best match is never dropped in favour of
    a worse one. When even the first chunk does not fit, it is cut to the budget
    rather than dropped, so a single large chunk still yields an answer.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    truncated = False

    for chunk in hits:
        block = format_block(len(blocks) + 1, chunk)
        cost = len(block) + len(BLOCK_SEPARATOR)
        if blocks and len(BLOCK_SEPARATOR.join(blocks)) + cost > max_chars:
            truncated = True
            break
        if not blocks and len(block) > max_chars:
            block = block[:max_chars]
            truncated = True
        blocks.append(block)
        used.append(chunk)

    text = BLOCK_SEPARATOR.join(blocks)
    return BuiltContext(text=text, used=tuple(used), chars=len(text), truncated=truncated)


def build_prompt(question: str, context_text: str) -> str:
    """Assemble the user message: context, then question, then the reminder."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    return (
        f"{CONTEXT_HEADER}\n"
        f"{context_text}\n\n"
        f"{QUESTION_HEADER}\n{question}\n\n"
        "Answer in at most three sentences using only the context above, and cite the "
        "block numbers you used. If the context does not contain the answer, reply with "
        f'"{INSUFFICIENT_CONTEXT_ANSWER}"'
    )


def estimate_prompt_tokens(prompt: str) -> int:
    """Rough token count for the prompt.

    A deliberately crude heuristic (English is around four characters per token)
    used only to warn in the logs when a prompt is close to the window. It is
    never used to trim content: ``build_context`` is the real budget.
    """
    return len(prompt) // 4 + 1


def prompt_fits_window(prompt: str, *, context_window: int, reserve_tokens: int = 512) -> bool:
    """Whether the prompt plus a reserved answer budget fits the model's window.

    The system prompt travels as a separate request field and the model needs
    room left over to generate, so part of the window is reserved rather than
    filled. A False result is logged as a warning -- it means the answer may be
    cut short -- but it never blocks the request.
    """
    return (
        estimate_prompt_tokens(SYSTEM_PROMPT)
        + estimate_prompt_tokens(prompt)
        + reserve_tokens
        <= context_window
    )
