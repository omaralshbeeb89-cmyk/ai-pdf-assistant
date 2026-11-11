import random

import pytest

from pdf_assistant.ingest import split_into_chunks, tokenize


def random_text(word_count, seed=0):
    rng = random.Random(seed)
    words = [
        "".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 12)))
        for _ in range(word_count)
    ]
    return " ".join(words), set(words)


def test_short_text_is_one_chunk():
    assert split_into_chunks("short text", 1000, 200) == ["short text"]


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_text_gives_no_chunks(text):
    assert split_into_chunks(text, 1000, 200) == []


def test_chunks_respect_max_size():
    text, _ = random_text(3000)
    chunks = split_into_chunks(text, 1000, 200)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_words_are_never_cut():
    text, words = random_text(3000)

    for chunk in split_into_chunks(text, 1000, 200):
        for word in chunk.split():
            assert word in words, f"cut word: {word!r}"


def test_consecutive_chunks_overlap():
    text, _ = random_text(3000)
    chunks = split_into_chunks(text, 1000, 200)

    for first, second in zip(chunks, chunks[1:]):
        # The start of each chunk repeats the end of the previous one.
        assert second.split()[0] in first.split()[-40:]


def test_all_text_is_covered():
    text, _ = random_text(3000)
    chunks = split_into_chunks(text, 1000, 200)

    covered = set(" ".join(chunks).split())
    assert covered == set(text.split())


def test_text_without_spaces_falls_back_to_hard_cuts():
    chunks = split_into_chunks("x" * 2500, 1000, 200)

    assert [len(chunk) for chunk in chunks] == [1000, 1000, 900]


def test_newlines_count_as_word_boundaries():
    text = "\n".join(["word"] * 500)

    for chunk in split_into_chunks(text, 100, 20):
        assert all(word == "word" for word in chunk.split())


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Ridge-Regression, (λ) RSS!") == ["ridge", "regression", "λ", "rss"]
