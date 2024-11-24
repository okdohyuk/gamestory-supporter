# RAG 기반 게임 개발 스토리 서포터

이 프로젝트는 **게임 개발 스토리**에 대한 질문-응답 시스템을 구축하는 **RAG(정보 검색 증강)** 기반의 **Streamlit** 웹 애플리케이션입니다. 사용자는 게임 관련 정보를 질문하고, **LangChain**, **Chroma**, **OpenAI API** 등을 활용하여 실시간으로 정확한 답변을 받을 수 있습니다.

## 📌 주요 기능

- **게임 개발 스토리** 관련 문서에서 실시간 질문 응답.
- **RAG 시스템**을 기반으로 한 질문-응답 (문서 검색 + GPT-4 생성).
- **Streamlit**을 활용한 직관적인 사용자 인터페이스 제공.
- **장르 육성의 걸작 조합** 등 다양한 게임 내 조건에 대한 질문 가능.

## 🛠️ 기술 스택

- **Streamlit**: 웹 애플리케이션 프레임워크
- **LangChain**: 텍스트 처리 및 검색 기반 모델
- **OpenAI GPT-4o**: 질문 응답 모델
- **Chroma**: 벡터 데이터베이스
- **OpenAIEmbeddings**: 텍스트 임베딩 처리

## 🖥️ 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/okdohyuk/gamestory-supporter.git
cd gamestory-supporter
```

### 2. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. Streamlit 실행

```bash
streamlit run gamestory.py
```

### 4. OpenAI API 키 설정

	•	OPENAI_API_KEY 환경 변수를 설정하여 OpenAI API 키를 입력합니다.
	•	export OPENAI_API_KEY="your_openai_api_key_here"

## 📝 사용 방법

	1.	웹 애플리케이션이 실행되면, 게임 개발 스토리와 관련된 질문을 입력할 수 있는 입력창이 표시됩니다.
	2.	예시 질문: "장르 육성의 걸작 조합리스트를 알려줘"
	3.	입력 후 “Submit” 버튼을 클릭하면, 실시간으로 관련 정보를 검색하고 GPT-4o 모델이 적절한 답변을 생성합니다.

## 🔧 기능 설명

### 1. 문서 로드 및 전처리

WebBaseLoader를 사용하여 지정된 URL로부터 게임 개발 관련 정보를 로드하고, RecursiveCharacterTextSplitter로 데이터를 적절한 크기로 분할하여 모델에 맞게 처리합니다.

### 2. 임베딩 및 벡터 저장소 생성

OpenAIEmbeddings를 사용하여 데이터를 임베딩하고, Chroma를 활용하여 벡터화된 데이터를 저장합니다.

### 3. 질문-응답 시스템

사용자가 입력한 질문에 대해 RAG 시스템을 통해 관련 데이터를 검색하고, GPT-4o 모델을 사용하여 최적의 답변을 생성합니다.