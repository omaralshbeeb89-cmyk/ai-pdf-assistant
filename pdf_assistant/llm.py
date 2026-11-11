"""Talking to the chat model: query rewriting and answer generation."""

import logging

from . import config


logger = logging.getLogger(__name__)


def format_history(history):
    """Convert conversation history into readable text."""

    if not history:
        return "No previous conversation."

    return "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in history
    )


def trim_history(history):
    """Keep only the most recent messages."""

    return history[-config.MAX_HISTORY_MESSAGES:]


def rewrite_question(question, history):
    """
    Convert a follow-up question into a standalone search query.
    Falls back to the original question if the rewrite fails.
    """

    if not history:
        return question

    prompt = f"""
You are a query rewriting assistant for a PDF question-answering system.

Your task is to rewrite the latest user question
into a clear standalone search query.

Use the conversation history only to understand
references such as:
- it
- they
- this concept
- explain more
- give me an example

Rules:
- Preserve the user's original meaning.
- Resolve references using conversation history.
- Do not answer the question.
- Do not add facts.
- Keep important technical terms.
- Return only the rewritten search query.
- If the question is already clear, return it unchanged.

CONVERSATION HISTORY:
{format_history(trim_history(history))}

LATEST USER QUESTION:
{question}

STANDALONE SEARCH QUERY:
"""

    try:
        response = config.get_client().responses.create(
            model=config.CHAT_MODEL,
            input=prompt,
        )
        rewritten = response.output_text.strip()

    except Exception as error:
        logger.warning("Query rewrite failed, using original question: %s", error)
        return question

    return rewritten or question


def build_answer_prompt(question, results, history):
    context = "\n\n---\n\n".join(
        f"[PDF page {result.page}]\n{result.text}"
        for result in results
    )

    return f"""
You are an intelligent PDF assistant.

Answer the user's latest question using the supplied PDF context.

You also receive previous conversation history.

IMPORTANT RULES:

1. Use conversation history to understand references
   and follow-up questions.

2. Use the PDF CONTEXT as the source of factual information.

3. Do not treat previous assistant answers as verified
   facts if they are not supported by the current context.

4. Do not invent information.

5. If the answer cannot be found in the PDF context,
   say:
   "I couldn't find the answer in the provided PDF."

6. Cite factual claims using [PDF page N].

7. Only cite page numbers explicitly shown in the context.

8. Explain concepts clearly and simply.

PREVIOUS CONVERSATION:
{format_history(trim_history(history))}

PDF CONTEXT:
{context}

LATEST USER QUESTION:
{question}

ANSWER:
"""


def stream_answer(question, results, history):
    """Yield the answer piece by piece as the model writes it."""

    stream = config.get_client().responses.create(
        model=config.CHAT_MODEL,
        input=build_answer_prompt(question, results, history),
        stream=True,
    )

    for event in stream:

        if event.type == "response.output_text.delta":
            yield event.delta

        elif event.type == "response.failed":
            error = getattr(event.response, "error", None)
            raise RuntimeError(f"The model failed to answer: {error}")

        elif event.type == "error":
            raise RuntimeError(f"The model failed to answer: {event.message}")
