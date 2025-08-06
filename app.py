# app.py

import streamlit as st
from rag_pipeline.loader import load_documents, extract_images_from_pdf
from rag_pipeline.chunker import chunk_by_method
from rag_pipeline.embedder import (
    CLIPImageEmbedder, CLIPTextEmbedder,
    get_text_embedder, get_image_embedder,
    get_hybrid_embedder
)
from rag_pipeline.es_uploader import (
    upload_image_embeddings_to_es, search_similar_images,
    upload_text_chunks_to_es
)
from rag_pipeline.retriever import search_similar_text
from rag_pipeline.reranker import rerank_cosine, rerank_crossencoder
from langchain.embeddings import HuggingFaceEmbeddings
from rag_pipeline.llm_engine import run_llm_qa

from langchain.llms import OpenAI  # 또는 KoAlpaca 등
import numpy as np


st.set_page_config(page_title="📚 RAG 실험실", layout="wide")
st.title("📄 PDF 기반 RAG 실험")

mode = st.sidebar.selectbox("모드 선택", ["text", "image_ocr", "image_embed"])
embedding_model = st.sidebar.selectbox("embedding 모델", {
    "text" : ["sbert", "openai", "jina"],
    "image_ocr" : ["sbert", "openai"],
    "image_embed" : ["clip", "siglip"]
}[mode])
chunking_method = None
if mode in ["text", "image_ocr"]:
    chunking_method = st.sidebar.selectbox("chunking 방법", ["recursive", "token","sentence"])
elif mode == "image_embed":
    st.sidebar.markdown("청킹 방법 : 없음 (이미지 임베딩 모드)")
retriever_method = st.sidebar.selectbox("retriever 방법", ["elasticsearch", "hybrid_es","hybrid_embedding","faiss"])
if mode == "image_embed":
    st.sidebar.markdown("rerank 방법 : 없음 (이미지 임베딩 모드)")
    rerank_method = "none"
else:
    rerank_method = st.sidebar.selectbox("rerank 방법", ["none", "cosine", "crossencoder"])
# rerank_method = st.sidebar.selectbox("rerank 방법", ["none", "cosine", "crossencoder"])


with st.sidebar.expander("🧪 현재 실험 조합 보기", expanded=True):
    st.markdown("**✅ 선택된 실험 설정**")
    st.write(f"🔹 모드 : `{mode}`")
    st.write(f"🔹 임베딩 모델: `{embedding_model}`")
    if chunking_method:
        st.write(f"🔹 청킹 방법: `{chunking_method}`")
    else:
        st.write("🔹 청킹 방법: 없음 (image embedding mode)")
    st.write(f"🔹 검색 방법 : `{retriever_method}`")
    st.write(f"🔹 리랭킹 방법: `{rerank_method}`")


def get_embedder(mode, model_name):
    if mode in ["text", "image_ocr"]:
        return get_text_embedder(model_name)
    elif mode == "image_embed":
        return get_image_embedder(model_name)
    elif mode == "text" and retriever_method == "hybrid_embedding":
        return get_hybrid_embedder()

if st.sidebar.button("📂 문서 인덱싱 실행"):
    # embeddr = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    # embeddr = get_embedder(mode, embedding_model)
    if retriever_method == "hybrid_embedding":
        text_embedder, image_embedder = get_hybrid_embedder()
    else:
        embeddr = get_embedder(mode, embedding_model)

    if mode in ["text", "image_ocr"]:
        st.info("🔍 문서 로딩 중...")
        raw_docs = load_documents(mode="text" if mode == "text" else "image")

        st.write("### 📄 문서별 텍스트 길이")
        for i, doc in enumerate(raw_docs):
            st.markdown(f"**문서 {i+1}: {doc['meta'].get('source', 'unknown')}**")
            st.write(f"문자 수: {len(doc['content'])}")
            st.code(doc['content'][:300])

        st.info(f"✂️ 청킹 시작... ({chunking_method} 방식)")
        chunks = chunk_by_method(raw_docs, method=chunking_method)

        st.success(f"✅ 총 {len(chunks)}개 청크 생성됨")
        st.write("### ✂️ 일부 청크 미리보기")
        for i, doc in enumerate(chunks[:3]):
            st.markdown(f"**Chunk {i+1}:**")
            st.code(doc.page_content[:500])

        st.info("🧠 임베딩 수행 중...")
        texts = [doc.page_content for doc in chunks]
        # vectors = embeddr.embed_documents(texts)
        if retriever_method == "hybrid_embedding":
            vectors = [text_embedder.embed_query(t) for t in texts]
        else:
            vectors = embeddr.embed_documents(texts)

        st.success(f"✅ 임베딩 완료 (vector shape: {len(vectors)} x {len(vectors[0])})")
        st.write("### 🔢 첫번째 임베딩 벡터 미리보기")
        st.code(str(vectors[0][:10]) + " ...")

        # 업로드
        # metadatas = [doc.metadata for doc in chunks]
        # if mode == "text":
        #     if retriever_method == "hybrid_es":
        #         index_name = "text_docs_hybrid"
        #     else:
        #         index_name = "text_docs"
        # elif mode == "image_ocr":
        #     index_name = "ocr_docs"
        # else:
        #     index_name = "image_embed_docs"

        index_name = "text_docs"
        if retriever_method == "hybrid_es":
            index_name = "text_docs_hybrid"
        elif mode == "image_ocr":
            index_name = "ocr_docs"

        # upload_text_chunks_to_es(chunks, embeddr, index_name=index_name)
        upload_text_chunks_to_es(chunks, text_embedder if retriever_method == "hybrid_embedding" else embeddr, index_name=index_name)
        st.success("✅ Elasticsearch 업로드 완료!")

        if retriever_method == "hybrid_embedding":
            from os import listdir
            from os.path import join
            all_images = []
            for fname in listdir("docs"):
                if fname.endswith(".pdf"):
                    images = extract_images_from_pdf(join("docs", fname))
                    all_images.extend(images)
            image_vectors = [image_embedder.embed_image(img) for img, _ in all_images]
            image_metas = [meta for _, meta in all_images]
            upload_image_embeddings_to_es(image_vectors, image_metas)
            st.success("✅ hybrid 이미지 벡터 업로드 완료!")

    # if mode == "text":
    #     raw_docs = load_documents(mode="text")
    #     chunks = chunk_by_method(raw_docs, method=chunking_method)
    #     upload_text_chunks_to_es(chunks, embeddr, index_name="text_docs")
    #     # embedder = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    #     # TODO: embed + upload to FAISS or Elasticsearch
    #     st.success("✅ 텍스트 임베딩 및 업로드 완료!")

    # elif mode == "image_ocr":
    #     raw_docs = load_documents(mode="image")
    #     chunks = chunk_by_method(raw_docs, method=chunking_method)
    #     upload_text_chunks_to_es(chunks, embeddr, index_name="ocr_docs")
    #     # embedder = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    #     # TODO: embed + upload to FAISS or Elasticsearch
    #     st.success("✅ 이미지 OCR 임베딩 및 업로드 완료!")

    elif mode == "image_embed":
        image_embedder = get_image_embedder(embedding_model)
        # all_images = []
        
        from os import listdir
        from os.path import join

        # st.info("🔍 PDF에서 이미지 추출 중...")

        # for fname in listdir("docs"):
        #     if fname.endswith(".pdf"):
        #         images = extract_images_from_pdf(join("docs", fname))
        #         all_images.extend(images)

        # st.success(f"✅ 총 {len(all_images)}개 이미지 추출됨")

        if "all_images" not in st.session_state:
            st.info("📄 PDF에서 이미지 추출 중...")
            all_images = []
            for fname in listdir("docs"):
                if fname.endswith(".pdf"):
                    images = extract_images_from_pdf(join("docs", fname))
                    all_images.extend(images)
            st.session_state.all_images = all_images
            st.success(f"✅ 총 {len(all_images)}개 이미지 추출됨")
        else:
            all_images = st.session_state.all_images
            st.success(f"✅ 이미지 캐시 로드 완료 (총 {len(all_images)}개)")

        # 이미지 샘플 미리보기 (Streamlit에서 PIL 이미지 보여주기)
        st.markdown("### 이미지 미리보기 (최대 3장)")
        for i, (img, meta) in enumerate(all_images[:3]):
            st.image(img, caption=f"{meta.get('source')} - p{meta.get('page')}", use_column_width=True)

        if "image_vectors" not in st.session_state:
            st.info("🧠 이미지 임베딩 중...")
            vectors = [image_embedder.embed_image(img) for img, _ in all_images]
            metadatas = [meta for _, meta in all_images]
            st.session_state.image_vectors = vectors
            st.session_state.image_metadatas = metadatas
            st.success(f"✅ 임베딩 완료 (vector dim: {len(vectors[0])})")
            st.code(f"첫 번째 벡터 preview: {vectors[0][:10]} ...")
            st.write("### 📝 예시 메타데이터:")
            st.json(metadatas[0])
            upload_image_embeddings_to_es(vectors, metadatas)
            st.success("✅ 이미지 임베딩 및 업로드 완료!")
        else:
            vectors = st.session_state.image_vectors
            metadatas = st.session_state.image_metadatas
            st.success("✅ (캐시) 임베딩 완료 및 업로드 완료됨")

        # st.info("🧠 이미지 임베딩 중...")

        # vectors = [image_embedder.embed_image(img) for img, _ in all_images]
        # metadatas = [meta for _, meta in all_images]

        # st.success(f"✅ 임베딩 완료 (vector dim: {len(vectors[0])})")
        # st.code(f"첫 번째 벡터 preview: {vectors[0][:10]} ...")

        # st.write("### 🔢 예시 메타데이터:")
        # st.json(metadatas[0])
        
        # upload_image_embeddings_to_es(vectors, metadatas)
        # st.success("✅ 이미지 임베딩 및 업로드 완료!")

query = st.text_input("질문을 입력하세요:")

if query:
    embedder = get_embedder(mode, embedding_model)
    
    # # 질의 임베딩
    # if hasattr(embedder, "embed_query"):
    #     query_vec = embedder.embed_query(query)
    # else:
    #     query_vec = embedder.embed_query(query)

    # image_embed 모드일 때는 text embedder 따로 처리
    if retriever_method == "hybrid_embedding":
        text_embedder, image_embedder = get_hybrid_embedder()
        query_vec = { #text_embedder.embed_query(query)
            "text": text_embedder.embed_query(query),
            "image": image_embedder.embed_text(query), # CLIPTextEmbedder or SigLIPTextEmbedder
        }
    
    elif mode == "image_embed":
        if embedding_model == "clip":
            query_embedder = CLIPTextEmbedder()
        elif embedding_model == "siglip":
            from rag_pipeline.embedder import SiglipTextEmbedder
            query_embedder = SiglipTextEmbedder()
        else:
            raise ValueError(f"지원하지 않는 임베딩 모델: {embedding_model}")
        query_vec = query_embedder.embed_text(query)
    
    else: # text or image_ocr
        embedder = get_embedder(mode, embedding_model)
        if hasattr(embedder, "embed_query"):
            query_vec = embedder.embed_query(query)
        else:
            query_vec = embedder.embed_text(query)

    # 검색
    from rag_pipeline.retriever import get_retriever
    retriever = get_retriever(mode, retriever_type=retriever_method)
    # hits = retriever(query_vec)

    if retriever_method == "hybrid_embedding":
        hits = retriever(query_vec) # query_vec is a dict with 'text' and 'image'
    elif retriever_method == "hybrid_es":
        hits = retriever(query, query_vec)
    else:
        hits = retriever(query_vec)

    # 문서 정리
    # docs = [
    #     {
    #         "content": hit["_source"]["content"],   
    #         "metadata": hit["_source"].get("metadata", {})
    #     }
    #     for hit in hits
    # ]
    # 문서 정리
    if mode == "image_embed":
        docs = [
            {
                # content는 없음
                "metadata": hit["_source"].get("metadata", {})
            }
            for hit in hits
        ]
    else:
        docs = [
            {
                "content": hit["_source"]["content"],
                "metadata": hit["_source"].get("metadata", {})
            }
            for hit in hits
        ]

    # rerank
    from rag_pipeline.reranker import rerank_documents
    # docs = rerank_documents(
    #     query=query,
    #     docs=docs,
    #     method=rerank_method,
    #     query_vec=query_vec,
    #     embedder=embedder,
    # )
    
    # 중복 제거
    seen_sources = set()
    unique_docs = []
    for doc in docs:
        src = doc["metadata"].get("source", "")  # PDF 파일명
        if src not in seen_sources:
            unique_docs.append(doc)
            seen_sources.add(src)
    
    # rerank_method가 none이 아니고 content가 있을 때만 reranking 수행
    if rerank_method != "none" and "content" in unique_docs[0]:
        docs = rerank_documents(
            query=query,
            docs=unique_docs,
            method=rerank_method,
            query_vec=query_vec,
            embedder=embedder,
        )
    else:
        docs = unique_docs

    # if rerank_method == "cosine":
    #     docs = rerank_cosine(query_vec, docs, embedder)

    #     st.markdown("### 🔍 유사 이미지 문서:")
    #     for i, hit in enumerate(hits):
    #         meta = hit["_source"]["metadata"]
    #         st.markdown(f"**{i+1}. {meta.get('source')} (p{meta.get('page')})**")
    
    # else: # text or image_ocr
    #     embedder = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    #     query_vec = embedder.embed_query(query)
    #     index_name = "text_docs" if mode == "text" else "ocr_docs"
    #     hits = search_similar_text(query_vec, index_name=index_name)

    #     docs = [
    #         {
    #             "content": hit["_source"]["content"],
    #             "metadata": hit["_source"].get("metadata", {})
    #         }
    #         for hit in hits
    #     ]

    #     if rerank_method == "cosine":
    #         docs = rerank_cosine(query_vec, docs, embedder)
    #     elif rerank_method == "cross-encoder":
    #         docs = rerank_crossencoder(query, docs)
    
    # 출력
    st.markdown("### 🔍 결과:")
    for i, doc in enumerate(unique_docs[:3]):
        # st.markdown(f"**{i+1}. {doc['metadata'].get('source', 'unknown')}**")
        # st.write(doc["content"][:500])
        meta = doc["metadata"]
        st.markdown(f"**{i+1}. {meta.get('source', 'unknown')} (p{meta.get('page', '-')})**")
    
        # 🔧 content가 있으면 보여주고, 없으면 "이미지 기반 문서"로 표시
        if "content" in doc:
            st.write(doc["content"][:500])
        else:
            st.info("📷 이미지 기반 문서입니다 (텍스트 없음)")

            # ✅ 이미지도 함께 출력
            if "all_images" in st.session_state:
                for img, img_meta in st.session_state.all_images:
                    if (
                        img_meta.get("source") == meta.get("source")
                        and img_meta.get("page") == meta.get("page")
                    ):
                        st.image(img, caption=f"{meta.get('source')} - p{meta.get('page')}")
                        break  # 가장 처음 매칭된 이미지만 출력

    from langchain.schema.document import Document as LCDocument
    if mode != "image_embed":
        llm = OpenAI(temperature=0.7)
        answer = run_llm_qa(query, [LCDocument(page_content=doc["content"],
                    metadata=doc["metadata"]) for doc in unique_docs[:3]], llm)
        st.markdown("### 💡 답변:")
        st.write(answer)
    else:
        st.info("📷 이미지 기반 문서는 텍스트가 없어 LLM 답변 생성을 생략합니다.")

# TODO: text/image_ocr → embedding → 검색 → rerank → run_llm_qa
