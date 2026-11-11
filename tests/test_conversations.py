import pytest

from pdf_assistant import config, conversations


@pytest.fixture(autouse=True)
def conversations_dir(tmp_path, monkeypatch):
    path = tmp_path / "conversations"
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", path)
    return path


def answered(question="What is the lasso?", cost=0.001):
    conversation = conversations.new_conversation(["C:/pdf/a.pdf"], ["hash-a"])
    conversation["messages"] = [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "A shrinkage method.",
            "sources": [],
            "usage": {"input_tokens": 1000, "output_tokens": 50, "cost": cost},
        },
    ]
    return conversation


def test_empty_conversation_is_not_saved(conversations_dir):
    conversations.save(conversations.new_conversation())

    assert not conversations_dir.exists() or not any(conversations_dir.iterdir())


def test_save_and_load_round_trip():
    conversation = answered()
    conversations.save(conversation)

    loaded = conversations.load(conversation["id"])

    assert loaded == conversation
    assert loaded["pdf_hashes"] == ["hash-a"]


def test_title_comes_from_first_question():
    conversation = answered("  What   is\nthe lasso?  ")
    conversations.save(conversation)

    assert conversation["title"] == "What is the lasso?"


def test_long_titles_are_shortened():
    conversation = answered("x" * 100)
    conversations.save(conversation)

    assert conversation["title"] == "x" * 60 + "…"


def test_list_is_newest_first_with_cost_and_question_count(monkeypatch):
    # A clock that moves forward one second per call.
    seconds = iter(range(10, 60))
    monkeypatch.setattr(conversations, "_now", lambda: f"2026-01-01T10:00:{next(seconds)}")

    old = conversations.new_conversation()
    old["messages"] = answered("old question", cost=0.002)["messages"]
    conversations.save(old)

    new = conversations.new_conversation()
    new["messages"] = answered("new question")["messages"]
    conversations.save(new)

    listed = conversations.list_conversations()

    assert [item["title"] for item in listed] == ["new question", "old question"]
    assert listed[1]["cost"] == pytest.approx(0.002)
    assert listed[0]["questions"] == 1


def test_unreadable_files_are_skipped(conversations_dir):
    conversations.save(answered())
    (conversations_dir / "broken.json").write_text("{not json", encoding="utf-8")

    assert len(conversations.list_conversations()) == 1


def test_delete_removes_the_file():
    conversation = answered()
    conversations.save(conversation)

    conversations.delete(conversation["id"])

    assert conversations.list_conversations() == []


@pytest.mark.parametrize("bad_id", ["../../secrets", "abc", "", None, "ABCDEF123456"])
def test_invalid_ids_are_refused(bad_id):
    with pytest.raises(ValueError):
        conversations.load(bad_id)


def test_history_keeps_only_role_and_content():
    assert conversations.history(answered()) == [
        {"role": "user", "content": "What is the lasso?"},
        {"role": "assistant", "content": "A shrinkage method."},
    ]


def test_conversation_usage_adds_up_answers():
    conversation = answered(cost=0.001)
    conversation["messages"] += answered(cost=0.002)["messages"]

    total = conversations.conversation_usage(conversation)

    assert total["cost"] == pytest.approx(0.003)
    assert total["input_tokens"] == 2000
