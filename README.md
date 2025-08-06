📄 PDF 문서
   ├── 텍스트 PDF ──→ 텍스트 추출 ──→ chunking ──→ 임베딩 ──→ Elasticsearch
   └── 이미지 PDF ──→ 이미지 추출 ──→ OCR ──→ chunking ──→ 임베딩 ──→ Elasticsearch

* 임베딩 부분은 아래를 뜻함
📄 PDF
 └─→ 이미지 추출 (page 단위)
       └─→ CLIP 임베딩
             └─→ Elasticsearch 저장 (index: image_embed_docs)

1. docs/ 폴더에 있는 PDF 문서를 불러오기
2. 텍스트 기반 또는 OCR 기반으로 텍스트 추출
3. chunking + embedding
4. Elasticsearch에 업로드
5. Streamlit UI로 질문 → 검색 → rerank → LLM 답변 생성
6. 다양한 실험 조합을 선택하고 결과 확인 가능

your_project/
├── app.py                          # Streamlit UI 메인 파일
├── docs/                           # PDF 문서 저장 폴더
├── rag_pipeline/
│   ├── __init__.py
│   ├── loader.py                   # 텍스트/이미지 로더
│   ├── chunker.py                  # chunking 알고리즘
│   ├── embedder.py                 # embedding 모델 래퍼
│   ├── es_uploader.py              # Elasticsearch 업로드/검색
│   ├── reranker.py                 # reranking (cosine, cross-encoder)
│   ├── llm_engine.py               # LLM을 통한 답변 생성
│   └── config.py                   # 설정값 공통 관리
└── requirements.txt                # 의존성 목록




#ver2.
rag_pipeline/
│
├── embedder/
│   ├── text_embedder.py     # sbert, openai 등
│   └── image_embedder.py    # clip, siglip
│
├── chunker/
│   └── chunker.py           # recursive / token
│
├── retriever/
│   └── retriever.py         # faiss / elasticsearch
│
├── reranker/
│   └── reranker.py          # cosine / crossencoder
│
├── ocr_engine.py            # PaddleOCR 등
├── loader.py                # PDF, 이미지 등 로딩
└── app.py                   # Streamlit UI