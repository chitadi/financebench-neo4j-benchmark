from fbneo.chunking import chunk_page_text
from fbneo.types import PageText


def test_chunk_page_text_overlaps() -> None:
    page = PageText(doc_name="doc", page_num=0, text=" ".join(f"w{i}" for i in range(20)))
    chunks = chunk_page_text(
        page,
        start_chunk_index=0,
        chunk_size_words=10,
        chunk_overlap_words=2,
    )
    assert len(chunks) == 3
    assert chunks[0][0] == 0
    assert chunks[1][0] == 1
    assert chunks[0][1].split()[-2:] == chunks[1][1].split()[:2]

