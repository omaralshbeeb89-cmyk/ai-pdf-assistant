"""
Save conversations as JSON files so they survive restarts.

A conversation is a plain dict:
    id, title, created, updated,
    pdf_paths, pdf_hashes,   the PDFs it was asked about
    messages                 [{role, content, ...}]; assistant messages
                             also hold sources, query, multi_doc, usage
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime

from . import config, usage


logger = logging.getLogger(__name__)

NEW_TITLE = "New conversation"
_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _path(conversation_id):
    # Ids are only ever created by new_conversation(); refuse anything
    # else so a crafted id can't point outside the folder.
    if not _ID_PATTERN.match(conversation_id or ""):
        raise ValueError(f"Invalid conversation id: {conversation_id!r}")

    return config.CONVERSATIONS_DIR / f"{conversation_id}.json"


def new_conversation(pdf_paths=(), pdf_hashes=()):
    now = _now()

    return {
        "id": uuid.uuid4().hex[:12],
        "title": NEW_TITLE,
        "created": now,
        "updated": now,
        "pdf_paths": list(pdf_paths),
        "pdf_hashes": list(pdf_hashes),
        "messages": [],
    }


def history(conversation):
    """Messages in the form the chat model needs."""

    return [
        {"role": message["role"], "content": message["content"]}
        for message in conversation["messages"]
    ]


def conversation_usage(conversation):
    return usage.total(message.get("usage") for message in conversation["messages"])


def save(conversation):
    """Write the conversation to disk. Empty conversations are not saved."""

    if not conversation["messages"]:
        return

    if conversation["title"] == NEW_TITLE:
        first_question = next(
            message["content"]
            for message in conversation["messages"]
            if message["role"] == "user"
        )
        title = " ".join(first_question.split())
        conversation["title"] = title[:60] + ("…" if len(title) > 60 else "")

    conversation["updated"] = _now()

    config.CONVERSATIONS_DIR.mkdir(exist_ok=True)
    path = _path(conversation["id"])

    # Write to a temporary file first so a crash can't leave half a file.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(conversation, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load(conversation_id):
    return json.loads(_path(conversation_id).read_text(encoding="utf-8"))


def delete(conversation_id):
    path = _path(conversation_id)

    if path.exists():
        path.unlink()


def list_conversations():
    """Summaries of saved conversations, most recently updated first."""

    if not config.CONVERSATIONS_DIR.is_dir():
        return []

    summaries = []

    for path in config.CONVERSATIONS_DIR.glob("*.json"):
        try:
            conversation = json.loads(path.read_text(encoding="utf-8"))
            summaries.append({
                "id": conversation["id"],
                "title": conversation["title"],
                "updated": conversation["updated"],
                "questions": sum(m["role"] == "user" for m in conversation["messages"]),
                "cost": conversation_usage(conversation)["cost"],
            })
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("Skipping unreadable conversation %s: %s", path.name, error)

    return sorted(summaries, key=lambda item: item["updated"], reverse=True)
