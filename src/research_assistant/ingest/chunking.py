"""Token-bounded, overlapping source-text slices with page provenance."""

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from research_assistant.ingest.ingestion import DocumentPage
from research_assistant.model_config import MODEL_NAME, MODEL_REVISION

# Keep existing tokenizer constant names compatible with callers.
TOKENIZER_NAME = MODEL_NAME
TOKENIZER_REVISION = MODEL_REVISION


@dataclass(frozen=True)
class Chunk:
    """An original-text slice; chunk_index is zero-based within an input page."""

    chunk_id: str
    document_id: str
    source: str
    page_number: int | None
    section: str | None
    chunk_index: int
    text: str


def load_tokenizer() -> PreTrainedTokenizerFast:
    """Load only the pinned fast tokenizer, downloading it on first use."""
    return AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True
    )


def _sections(text: str) -> Iterator[tuple[str | None, str]]:
    """Split at ATX headings outside fenced code; keep all original characters."""
    start = offset = 0
    section = None
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                if not marker[2].strip():
                    fence = None
        elif marker:
            fence = marker[1]
        else:
            heading = re.match(
                r"^ {0,3}#{1,6}(?:[ \t]+(.*)|[ \t]*)$", line.rstrip("\r\n")
            )
            if heading:
                if offset > start:
                    yield section, text[start:offset]
                start = offset
                section = re.sub(r"[ \t]+#+[ \t]*$", "", heading[1] or "").strip()
        offset += len(line)
    if start < len(text):
        yield section, text[start:]


def _windows(
    text: str, tokenizer: PreTrainedTokenizerFast, max_tokens: int, overlap: int
) -> Iterator[str]:
    budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    start = 0
    while start < len(text):
        remaining = text[start:]
        if not remaining.strip():
            return
        offsets = tokenizer(
            remaining,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )["offset_mapping"]
        end_token = min(budget, len(offsets))
        while True:
            end = offsets[end_token][0] if end_token < len(offsets) else len(remaining)
            candidate = remaining[:end]
            count = len(tokenizer.encode(candidate, truncation=False, verbose=False))
            if end > 0 and candidate.strip() and count <= max_tokens:
                break
            end_token -= 1
            if end_token <= 0:
                raise ValueError("Cannot fit a nonempty source span in max_tokens")
        yield candidate
        if end == len(remaining):
            return
        # Source token overlap may differ after independently encoding a subword
        # boundary. Always advance, even for unusually small final windows.
        next_token = max(1, end_token - overlap)
        advance = offsets[next_token][0]
        if advance <= 0:
            raise ValueError("Tokenizer offsets do not permit forward progress")
        start += advance


def chunk_pages(
    pages: Iterable[DocumentPage],
    *,
    max_tokens: int = 180,
    overlap: int = 30,
    tokenizer: PreTrainedTokenizerFast | None = None,
) -> list[Chunk]:
    """Chunk pages independently; max_tokens includes model special tokens.

    Overlap counts source tokens, resets at page/Markdown section boundaries,
    and may shrink when needed for progress. Supplied tokenizers must be fast.
    """
    if type(max_tokens) is not int or type(overlap) is not int:
        raise ValueError("max_tokens and overlap must be integers")
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("Require max_tokens > 0 and 0 <= overlap < max_tokens")
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer()
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= 0 or overlap >= budget:
        raise ValueError("Overlap must be smaller than the positive text-token budget")
    if max_tokens > tokenizer.model_max_length:
        raise ValueError("max_tokens exceeds the tokenizer model_max_length")
    tokenizer_config = json.loads(tokenizer.backend_tokenizer.to_str())
    # Calls mutate these execution settings; they are not tokenizer identity.
    tokenizer_config.pop("truncation", None)
    tokenizer_config.pop("padding", None)
    fingerprint = hashlib.sha256(
        json.dumps(tokenizer_config, sort_keys=True).encode()
    ).hexdigest()
    chunks = []
    for page in pages:
        sections = (
            _sections(page.text) if page.page_number is None else [(None, page.text)]
        )
        index = 0
        for section, text in sections:
            for window in _windows(text, tokenizer, max_tokens, overlap):
                identity = [
                    "chunk-v1",
                    fingerprint,
                    max_tokens,
                    overlap,
                    page.document_id,
                    page.source,
                    page.page_number,
                    section,
                    index,
                    window,
                ]
                chunk_id = hashlib.sha256(
                    json.dumps(identity, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                chunks.append(
                    Chunk(
                        chunk_id,
                        page.document_id,
                        page.source,
                        page.page_number,
                        section,
                        index,
                        window,
                    )
                )
                index += 1
    return chunks
