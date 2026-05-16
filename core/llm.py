import re
from functools import lru_cache

from utils.helpers import build_cited_context
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

GENERATION_MODEL = (
    "google/flan-t5-small"
)
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 160
MIN_ANSWER_CHARS = 30

# =========================
# Load Generator
# =========================

@lru_cache(maxsize=1)
def get_generator():

    tokenizer = (
        AutoTokenizer.from_pretrained(
            GENERATION_MODEL
        )
    )

    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            GENERATION_MODEL
        )
    )

    return tokenizer, model

# =========================
# Build Prompt
# =========================

def build_prompt(
    question,
    retrieved_docs,
):

    context = build_cited_context(
        retrieved_docs
    )

    if not context:
        context = "No supporting evidence was provided."

    return f"""
Answer the question using only the evidence below.
Write a concise, complete answer in 1 to 4 sentences.
Add a citation after each factual sentence, like [Source 1].
Use the source number from the evidence line.
Do not output only a source label.
If the evidence does not answer the question, say that the documents do not provide enough information.

Evidence:
{context}

Question: {question}

Answer:
""".strip()


def _tokenizer_max_length(tokenizer):
    model_max_length = getattr(
        tokenizer,
        "model_max_length",
        MAX_INPUT_TOKENS,
    )

    if not isinstance(model_max_length, int) or model_max_length > 100000:
        return MAX_INPUT_TOKENS

    return max(
        128,
        min(
            model_max_length,
            MAX_INPUT_TOKENS,
        ),
    )


def _is_citation_only(text):
    compact_text = re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()

    if not compact_text:
        return True

    return bool(
        re.fullmatch(
            r"(?:\[?Source\s+\d+\]?[.,;:\s]*)+",
            compact_text,
            flags=re.IGNORECASE,
        )
    )


def _clean_answer(text):
    answer = (
        text or ""
    ).strip()

    answer = re.sub(
        r"^(answer|final answer)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()

    answer = re.sub(
        r"\s+",
        " ",
        answer,
    ).strip()

    return answer


def _question_terms(question):
    terms = re.findall(
        r"[A-Za-z0-9]{4,}",
        question or "",
    )

    return {
        term.lower()
        for term in terms
    }


def _best_evidence_sentence(question, retrieved_docs):
    terms = _question_terms(
        question
    )
    best_sentence = ""
    best_source = 1
    best_score = -1

    for source_index, doc in enumerate(retrieved_docs or [], start=1):
        content = getattr(
            doc,
            "page_content",
            "",
        )
        sentences = re.split(
            r"(?<=[.!?])\s+",
            content,
        )

        for sentence in sentences:
            cleaned_sentence = re.sub(
                r"\s+",
                " ",
                sentence,
            ).strip()

            if len(cleaned_sentence) < 20:
                continue

            sentence_terms = set(
                re.findall(
                    r"[A-Za-z0-9]{4,}",
                    cleaned_sentence.lower(),
                )
            )
            score = len(
                terms & sentence_terms
            )

            if score > best_score:
                best_sentence = cleaned_sentence
                best_source = source_index
                best_score = score

    if not best_sentence:
        return (
            "The documents do not provide enough information to answer this question."
        )

    if len(best_sentence) > 240:
        best_sentence = best_sentence[:240].rsplit(
            " ",
            1,
        )[0].strip()

    if not best_sentence.endswith((".", "?", "!")):
        best_sentence += "."

    return f"{best_sentence} [Source {best_source}]"


def _stabilize_answer(answer, question, retrieved_docs):
    clean_answer = _clean_answer(
        answer
    )

    if (
        len(clean_answer) < MIN_ANSWER_CHARS
        or _is_citation_only(clean_answer)
    ):
        return _best_evidence_sentence(
            question,
            retrieved_docs,
        )

    if "[Source" not in clean_answer:
        clean_answer = f"{clean_answer} [Source 1]"

    return clean_answer

# =========================
# Generate Answer
# =========================

def generate_answer(
    question,
    retrieved_docs,
):

    prompt = build_prompt(
        question,
        retrieved_docs,
    )

    tokenizer, model = (
        get_generator()
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_tokenizer_max_length(
            tokenizer
        ),
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=1,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return _stabilize_answer(
        generated_text,
        question,
        retrieved_docs,
    )
