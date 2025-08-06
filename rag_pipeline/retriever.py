# rag_pipeline/retriever.py

# image_ocr → "ocr_docs"로 바꾸면 됨
# 결과는 chunk와 메타데이터 포함된 ES hit 목록 반환됨

from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def search_similar_text(query_vec, index_name="text_docs", top_k=5):
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": query_vec}
                }
            }
        }
    }
    return es.search(index=index_name, body=body)["hits"]["hits"]


def search_similar_images(query_vec, index_name="image_embed_docs", top_k=5):
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": query_vec.tolist()}
                }
            }
        }
    }
    return es.search(index=index_name, body=body)["hits"]["hits"]


def search_hybrid_es(query_text, query_vec, index_name="text_docs_hybrid", top_k=5):
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "match": {
                        "content": query_text
                    }
                },
                "script": {
                    "source": "0.5 * cosineSimilarity(params.query_vector, 'vector') + 0.5 * _score",
                    "params": {"query_vector": query_vec}
                }
            }
        }
    }
    return es.search(index=index_name, body=body)["hits"]["hits"]

# 요건 hybrid embedding에서 사용하기 위한 함수
def search_hybrid(query_text_vec, query_image_vec, top_k=5):
    text_hits = search_similar_text(query_text_vec, index_name="text_docs", top_k=top_k)
    image_hits = search_similar_text(query_image_vec, index_name="image_embed_docs", top_k=top_k)

    # 간단히 score 합산해서 정렬 (가중치 조정 가능)
    combined = {}

    for hit in text_hits:
        id_ = hit["_id"]
        combined[id_] = {"doc": hit, "score": hit["_score"] * 0.7}

    for hit in image_hits:
        id_ = hit["_id"]
        if id_ in combined:
            combined[id_]["score"] += hit["_score"] * 0.3
        else:
            combined[id_] = {"doc": hit, "score": hit["_score"] * 0.3}

    # 정렬 후 반환
    sorted_hits = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [x["doc"] for x in sorted_hits[:top_k]]


def get_retriever(mode: str, retriever_type: str = "elasticsearch"):
    """
    mode: text / image_ocr / image_embed
    retriever_type: elasticsearch (dense), hybrid (keyword+vector)
    """

    if retriever_type == "hybrid_es":
        if mode != "text":
            raise ValueError("hybrid_es retriever는 text 모드에서만 사용 가능합니다.")

        index_name = "text_docs_hybrid"

        def retriever(query_text, query_vector, top_k=5):
            return search_hybrid_es(query_text, query_vector, index_name=index_name, top_k=top_k)

        return retriever

    elif retriever_type == "hybrid_embedding":
        def retriever(query_embedding, top_k=5):
            return search_hybrid(query_text_vec=query_vec["text"], 
            query_image_vec=query_vec["image"], top_k=top_k)

        return retriever
    
    elif retriever_type == "elasticsearch":

    # if retriever_type != "elasticsearch":
    #     raise NotImplementedError("현재는 elasticsearch만 지원됩니다.")

        # 인덱스 이름 지정
        if mode == "text":
            index_name = "text_docs"
        elif mode == "image_ocr":
            index_name = "ocr_docs"
        elif mode == "image_embed":
            index_name = "image_embed_docs"
        else:
            raise ValueError(f"지원하지 않는 모드: {mode}")

        # 내부 함수 형태로 retriever 리턴
        def retriever(query_vector, top_k=5):
            return search_similar_text(query_vector, index_name=index_name, top_k=top_k)

        return retriever

    else:
        raise NotImplementedError(f"지원하지 않는 retriever 타입: {retriever_type}")