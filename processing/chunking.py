from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

def split_documents(
    documents,
    chunk_size,
    chunk_overlap,
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(
        chunks,
        start=1
    ):
        chunk.metadata["chunk"] = i

    return chunks