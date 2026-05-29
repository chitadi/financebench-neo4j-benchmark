from fbneo.data import question_document_info
from fbneo.types import DocumentMeta, EvidenceGold, FinanceBenchQuestion


def test_question_document_info_keeps_only_question_and_evidence_docs() -> None:
    questions = [
        FinanceBenchQuestion(
            financebench_id="q1",
            question="question",
            answer="answer",
            doc_name="doc_a",
            company="Company A",
            evidence=[EvidenceGold(doc_name="doc_b", evidence_page_num=1)],
        )
    ]
    docs = {
        "doc_a": DocumentMeta(doc_name="doc_a", company="Company A", doc_type="10-K"),
        "doc_b": DocumentMeta(doc_name="doc_b", company="Company B", doc_type="10-Q"),
        "unused_doc": DocumentMeta(doc_name="unused_doc", company="Company C"),
    }

    selected = question_document_info(questions, docs)

    assert set(selected) == {"doc_a", "doc_b"}
    assert selected["doc_a"].doc_type == "10-K"
    assert selected["doc_b"].doc_type == "10-Q"
