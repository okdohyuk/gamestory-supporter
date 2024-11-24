import streamlit as st
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# OpenAI API Key 환경 변수 설정
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# Streamlit 페이지 설정
st.set_page_config(page_title="게임 개발 스토리 서포터", layout="wide")

# Streamlit 제목
st.title("게임 개발 스토리 서포터")
st.markdown("### 게임 관련 정보를 질문하면, 실시간으로 답변을 생성합니다!")

# 데이터 로드 및 전처리
@st.cache_data
def load_and_process_documents():
    urls = [
        "https://namu.wiki/w/%EA%B2%8C%EC%9E%84%EA%B0%9C%EB%B0%9C%20%EC%8A%A4%ED%86%A0%EB%A6%AC",
        "https://namu.wiki/w/%EA%B2%8C%EC%9E%84%EA%B0%9C%EB%B0%9C%20%EC%8A%A4%ED%86%A0%EB%A6%AC/%EA%B4%80%EB%A0%A8%20%EC%A0%95%EB%B3%B4",
        "https://namu.wiki/w/%EA%B2%8C%EC%9E%84%EA%B0%9C%EB%B0%9C%20%EC%8A%A4%ED%86%A0%EB%A6%AC/%EA%B4%80%EB%A0%A8%20%EC%A0%95%EB%B3%B4/%EC%9E%A5%EB%A5%B4%20%EC%A1%B0%ED%95%A9",
        "https://namu.wiki/w/%EA%B2%8C%EC%9E%84%EA%B0%9C%EB%B0%9C%20%EC%8A%A4%ED%86%A0%EB%A6%AC/%EA%B4%80%EB%A0%A8%20%EC%A0%95%EB%B3%B4/%EB%82%B4%EC%9A%A9%20%EB%B0%9C%EA%B2%AC%20%EC%A1%B0%EA%B1%B4"
    ]

    # 1. 문서 로드
    loader = WebBaseLoader(
        web_path=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("wiki-heading-content", "wiki-paragraph")
            )
        ),
    )
    print("문서 로드 중...1")
    documents = loader.load()
    print("문서 로드 중...2")
    print(documents[1])

    # 2. 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    return split_docs

# 데이터 임베딩 및 벡터 저장소 생성
@st.cache_resource
def setup_vector_store(_docs):
    embeddings = OpenAIEmbeddings()

    # 데이터 저장 디렉토리 명시
    persist_directory = "./chroma_db"

    # ChromaDB 초기화 및 데이터 저장
    vector_store = Chroma.from_documents(_docs, embeddings, persist_directory=persist_directory)
    return vector_store

# QA 체인 구성
@st.cache_resource
def setup_qa_chain(_vector_store):
    retriever = _vector_store.as_retriever()

    # LLM 및 프롬프트 설정
    llm = ChatOpenAI(model_name="gpt-4o")
    prompt_template = """
    You are a helpful assistant for answering questions about Game Development Story. 
    Use the following context to answer the question: {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # RetrievalQA 체인 구성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # stuff, map_reduce 등 선택 가능
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# 데이터 로드
with st.spinner("문서를 로드하고 있습니다..."):
    documents = load_and_process_documents()

# 벡터 저장소 생성
with st.spinner("벡터 저장소를 준비 중입니다..."):
    vector_store = setup_vector_store(documents)

# QA 체인 생성
with st.spinner("QA 시스템을 준비 중입니다..."):
    qa_chain = setup_qa_chain(vector_store)

# 사용자 입력
user_question = st.text_input(
    "게임 개발 스토리에 대해 궁금한 점을 입력하세요:",
    placeholder="장르 육성의 걸작 조합리스트를 알려줘"
)

# 답변 생성
if user_question:
    with st.spinner("답변을 생성 중입니다..."):
        response = qa_chain.run(user_question)
        st.markdown("### 답변:")
        st.write(response)