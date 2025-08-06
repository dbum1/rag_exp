# rag_pipeline/chunker.py

import kss
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.schema.document import Document

def chunk_documents(docs, method="recursive", chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for doc in docs:
        splits = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(splits):
            meta = doc["meta"].copy()
            meta["chunk_id"] = i
            chunks.append(Document(page_content=chunk, metadata=meta))
    return chunks

def chunk_by_method(docs, method="recursive", chunk_size=500, chunk_overlap=50):
    """
    주어진 문서 리스트에 대해 청킹 방법(method)에 따라 문서를 나눔.
    - method: "recursive" | "token" | "kss"
    - docs: [{"content": str, "meta": dict}]
    - return: List[Document]
    """

    chunks = []

    for doc in docs:
        content = doc["content"]
        meta = doc["meta"]

        if method == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            splits = splitter.split_text(content)

        elif method == "token":
            splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            splits = splitter.split_text(content)

        elif method == "sentence":
            # 1.  문장 분리
            sentences = kss.split_sentences(content)
            # 2. 문장 누적하여 chunk_size 넘지 않게 chunking
            splits = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    splits.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk:
                splits.append(current_chunk.strip())

        else:
            raise ValueError(f"지원하지 않는 청킹 방식: {method}")


        for i, chunk in enumerate(splits):
            new_meta = meta.copy()
            new_meta["chunk_id"] = i
            chunks.append(Document(page_content=chunk, metadata=new_meta))

    return chunks
