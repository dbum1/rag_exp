# rag_pipeline/llm_engine.py

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

def run_llm_qa(query, docs, llm):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""다음 문서를 참고하여 질문에 한국어로 답해주세요.
문서에 없는 내용은 답하지 마세요.

문서: {context}

질문: {question}
답변:"""
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    qa_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    return qa_chain.invoke({
        "input_documents": docs,
        "question": query
    })["output_text"]
