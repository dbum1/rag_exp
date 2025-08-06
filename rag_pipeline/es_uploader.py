# rag_pipeline/es_uploader.py

from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# def upload_image_embeddings_to_es(image_embeds, metadatas, index_name="image_embed_docs"):
#     for vec, meta in zip(image_embeds, metadatas):
#         doc = {
#             "vector": vec, #vec.tolist(),
#             "metadata": meta
#         }
#         es.index(index=index_name, body=doc)

def upload_image_embeddings_to_es(image_embeds, metadatas, index_name="image_embed_docs"):
    for vec, meta in zip(image_embeds, metadatas):
        doc = {
            "vector": vec,  # vec.tolist()도 가능
            "metadata": meta,
            "content": ""  # 또는 이미지에 대응되는 설명이 있다면 그걸 넣기
        }
        es.index(index=index_name, body=doc)

# text_docs / ocr_docs
def upload_text_chunks_to_es(chunks, embedder, index_name="text_docs"):
    from elasticsearch import Elasticsearch
    es = Elasticsearch("http://localhost:9200")

    for chunk in chunks:
        vec = embedder.embed_query(chunk.page_content)
        doc = {
            "vector": vec, #vec.tolist(),
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        es.index(index=index_name, body=doc)



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