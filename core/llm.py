from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

GENERATION_MODEL = (
    "google/flan-t5-small"
)

# =========================
# Load Generator
# =========================

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

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    return f"""
Answer the question using only the context.

If answer not found say:
Answer not available in provided documents.

Context:
{context}

Question:
{question}

Answer:
"""

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
        max_length=1024,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=220,
        num_beams=4,
        do_sample=False,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()