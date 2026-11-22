"""Offline chunking checks plus an explicitly enabled real-tokenizer check."""

import os
from dataclasses import FrozenInstanceError, replace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from research_assistant.ingest.chunking import chunk_pages, load_tokenizer
from research_assistant.ingest.ingestion import DocumentPage


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    words = ["[UNK]", "[CLS]", "[SEP]", "hello", "world", "play", "##ing"]
    words += [f"w{i}" for i in range(300)]
    backend = Tokenizer(
        WordPiece({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=512,
    )


def page(text: str, number: int | None = 1) -> DocumentPage:
    return DocumentPage("document", "paper.pdf" if number else "notes.md", number, text)


def test_empty(tokenizer: PreTrainedTokenizerFast) -> None:
    assert chunk_pages([], tokenizer=tokenizer) == []
    assert chunk_pages([page(""), page(" \n\t")], tokenizer=tokenizer) == []


def test_short_and_immutable(tokenizer: PreTrainedTokenizerFast) -> None:
    (chunk,) = chunk_pages([page("  hello world\n")], tokenizer=tokenizer)
    assert chunk.text == "  hello world\n"
    assert (chunk.document_id, chunk.source, chunk.page_number, chunk.section) == (
        "document",
        "paper.pdf",
        1,
        None,
    )
    assert chunk.chunk_index == 0
    with pytest.raises(FrozenInstanceError):
        chunk.text = "changed"


def test_overlap_and_limits(tokenizer: PreTrainedTokenizerFast) -> None:
    text = " ".join(f"w{i}" for i in range(40))
    chunks = chunk_pages([page(text)], max_tokens=12, overlap=3, tokenizer=tokenizer)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for left, right in zip(chunks, chunks[1:]):
        assert left.text.split()[-3:] == right.text.split()[:3]
    for chunk in chunks:
        assert chunk.text in text
        assert chunk.text.strip()
        assert len(tokenizer.encode(chunk.text)) <= 12
    assert chunks[-1].text.endswith("w39")


def test_zero_overlap_reconstructs_source(tokenizer: PreTrainedTokenizerFast) -> None:
    text = " HéLLo\tplaying!!!\n" * 30
    chunks = chunk_pages([page(text)], max_tokens=10, overlap=0, tokenizer=tokenizer)
    assert "".join(c.text for c in chunks) == text
    assert all(len(tokenizer.encode(c.text)) <= 10 for c in chunks)


def test_subword_boundaries(tokenizer: PreTrainedTokenizerFast) -> None:
    text = "playing " * 12
    chunks = chunk_pages([page(text)], max_tokens=5, overlap=1, tokenizer=tokenizer)
    assert len(chunks) > 1
    assert all(c.text in text and "##" not in c.text for c in chunks)
    assert all(len(tokenizer.encode(c.text)) <= 5 for c in chunks)


def test_page_boundaries(tokenizer: PreTrainedTokenizerFast) -> None:
    chunks = chunk_pages([page("hello", 1), page("world", 3)], tokenizer=tokenizer)
    assert [(c.page_number, c.text, c.chunk_index) for c in chunks] == [
        (1, "hello", 0),
        (3, "world", 0),
    ]
    assert len({c.chunk_id for c in chunks}) == 2


def test_markdown_sections_and_fences(tokenizer: PreTrainedTokenizerFast) -> None:
    text = (
        "Intro\n# First\nhello\n```python\n# not a heading\n```\n## Second ##\nworld\n"
    )
    chunks = chunk_pages([page(text, None)], tokenizer=tokenizer)
    assert [c.section for c in chunks] == [None, "First", "Second"]
    assert "# not a heading" in chunks[1].text
    assert "".join(c.text for c in chunks) == text
    assert all(c.page_number is None and c.source == "notes.md" for c in chunks)


def test_ids(tokenizer: PreTrainedTokenizerFast) -> None:
    pages = [page("hello world " * 40)]
    chunks = chunk_pages(pages, max_tokens=12, overlap=2, tokenizer=tokenizer)
    assert chunks == chunk_pages(pages, max_tokens=12, overlap=2, tokenizer=tokenizer)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for changed in [
        replace(pages[0], source="other.pdf"),
        replace(pages[0], document_id="other"),
    ]:
        assert (
            chunk_pages([changed], tokenizer=tokenizer)[0].chunk_id
            != chunks[0].chunk_id
        )
    assert (
        chunk_pages(pages, max_tokens=12, overlap=1, tokenizer=tokenizer)[0].chunk_id
        != chunks[0].chunk_id
    )


@pytest.mark.parametrize(
    "size,overlap",
    [
        (0, 0),
        (-1, 0),
        (2, 0),
        (10, -1),
        (10, 8),
        (10, 10),
        (513, 0),
        (10.5, 1),
        (10, True),
    ],
)
def test_invalid_configuration(
    tokenizer: PreTrainedTokenizerFast, size: int, overlap: int
) -> None:
    with pytest.raises(ValueError):
        chunk_pages(
            [page("hello")], max_tokens=size, overlap=overlap, tokenizer=tokenizer
        )


@pytest.mark.skipif(
    os.environ.get("RUN_TOKENIZER_INTEGRATION") != "1",
    reason="Opt-in tokenizer download",
)
def test_actual_minilm_tokenizer() -> None:
    tokenizer = load_tokenizer()
    text = "Résumé: Attention(x) = softmax(QKᵀ / √d). Playing with subwords!\n" * 90
    chunks = chunk_pages([page(text)], tokenizer=tokenizer)
    assert len(chunks) > 1
    assert all(0 < len(tokenizer.encode(c.text)) <= 180 for c in chunks)
    assert all(c.text in text for c in chunks)
    assert chunks == chunk_pages([page(text)], tokenizer=tokenizer)
    nonoverlapping = chunk_pages([page(text)], overlap=0, tokenizer=tokenizer)
    assert "".join(c.text for c in nonoverlapping) == text
    assert tokenizer.num_special_tokens_to_add(pair=False) == 2


def test_ids_ignore_previous_padding_and_truncation(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    tokenizer.backend_tokenizer.enable_truncation(max_length=8)
    tokenizer.backend_tokenizer.enable_padding(length=8)
    first = chunk_pages([page("hello world")], tokenizer=tokenizer)
    assert first == chunk_pages([page("hello world")], tokenizer=tokenizer)
    changed = chunk_pages([page("hello world")], max_tokens=179, tokenizer=tokenizer)
    assert first[0].chunk_id != changed[0].chunk_id


def test_long_markdown_never_crosses_sections(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    text = "# One\n" + "hello " * 40 + "\n# Two\n" + "world " * 40
    chunks = chunk_pages(
        [page(text, None)], max_tokens=12, overlap=3, tokenizer=tokenizer
    )
    assert {c.section for c in chunks} == {"One", "Two"}
    assert all(not ("hello" in c.text and "world" in c.text) for c in chunks)
    assert all(len(tokenizer.encode(c.text)) <= 12 for c in chunks)
