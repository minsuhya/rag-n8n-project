# RAG + N8N + LMSTUDIO + StreamLit 프로젝트

전체 서비스(n8n, ChromaDB, FastAPI, Streamlit, LM Studio)를 하나의 `docker-compose.yml` 파일로 통합하고, 필요한 디렉토리 구조와 관련 코드를 제공하겠습니다. 이 구현은 MkDocs 문서를 기반으로 한 RAG 시스템을 실행하며, `process_documents`와 `embed` 엔드포인트를 포함한 FastAPI 서버, n8n 워크플로우, ChromaDB 벡터 데이터베이스, Streamlit UI를 통합합니다. LM Studio는 Docker가 아닌 호스트에서 실행되므로, `docker-compose.yml`에는 포함되지 않지만 네트워크 설정을 통해 통합됩니다.

---

## **1. 전체 아키텍처 개요**

- **서비스 구성**:
  - **n8n**: 워크플로우 자동화 (포트: 5678).
  - **ChromaDB**: 벡터 데이터베이스 (포트: 8001).
  - **FastAPI**: 문서 처리 및 임베딩 API (포트: 8000).
  - **Streamlit**: 사용자 인터페이스 (포트: 8501).
  - **LM Studio**: 로컬 LLM (호스트에서 실행, 포트: 1234).
- **데이터 흐름**:
  1. FastAPI가 MkDocs 문서를 처리하여 `./chroma_db`에 저장.
  2. Streamlit에서 사용자 질문이 n8n 워크플로우로 전송.
  3. n8n이 FastAPI의 `/embed`로 질문 임베딩 생성, ChromaDB에서 문서 검색, LM Studio로 답변 생성.
  4. Streamlit이 답변을 표시.
- **디렉토리 구조**: 모든 데이터와 코드를 체계적으로 관리.

---

## **2. 디렉토리 구조**

```
rag-project/
├── docker-compose.yml
├── chroma_db/                    # ChromaDB 데이터 저장소
├── my-docs/                      # MkDocs 문서
│   └── docs/
│       └── sample.md
├── api/                          # FastAPI 애플리케이션
│   ├── api.py
│   └── requirements.txt
├── streamlit/                    # Streamlit 애플리케이션
│   ├── app.py
│   └── requirements.txt
├── .env                          # 환경 변수
└── .gitignore
```

### **설명**

- **`chroma_db/`**: ChromaDB의 벡터 데이터 저장소.
- **`my-docs/docs/`**: MkDocs 문서 디렉토리 (예: `sample.md`).
- **`api/`**: FastAPI 코드와 의존성.
- **`streamlit/`**: Streamlit 코드와 의존성.
- **`.env`**: n8n 및 기타 서비스의 환경 변수.
- **`.gitignore`**: 불필요한 파일 제외.

---

## **3. 코드 및 설정**

### **3.1. `docker-compose.yml`**

모든 서비스를 정의하고, `./chroma_db`와 `./my-docs/docs`를 볼륨으로 공유합니다.

```yaml
version: "3.8"

services:
  n8n:
    image: n8nio/n8n:latest
    ports:
      - "5678:5678"
    environment:
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
    volumes:
      - ./chroma_db:/data/chroma_db
      - ./my-docs/docs:/data/shared/my-docs/docs
    networks:
      - rag-network
    depends_on:
      - chromadb
      - fastapi

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8001"
    command: ["--path", "/chroma_db", "--host", "0.0.0.0", "--port", "8001"]
    volumes:
      - ./chroma_db:/chroma_db
    networks:
      - rag-network

  fastapi:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./my-docs/docs:/app/my-docs/docs
    networks:
      - rag-network
    depends_on:
      - chromadb

  streamlit:
    build:
      context: ./streamlit
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    networks:
      - rag-network
    depends_on:
      - n8n

networks:
  rag-network:
    driver: bridge
```

**설명**:

- **n8n**: 포트 5678에서 실행, `./chroma_db`와 `./my-docs/docs`를 볼륨으로 마운트.
- **chromadb**: HTTP 서버 모드로 실행, `./chroma_db`에 데이터 저장.
- **fastapi**: `./api` 디렉토리에서 빌드, 문서와 ChromaDB 데이터 접근.
- **streamlit**: `./streamlit` 디렉토리에서 빌드, n8n과 통신.
- **networks**: 모든 서비스가 `rag-network`를 통해 통신.

---

### **3.2. `.env` 파일**

환경 변수를 정의합니다.

N8N_HOST=localhost
N8N_PORT=5678
N8N_PROTOCOL=http

---

### **3.3. FastAPI 애플리케이션**

`process_documents`와 `embed` 엔드포인트를 포함한 FastAPI 서버입니다.

#### **3.3.1. `api/api.py`**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class DocumentInput(BaseModel):
    file_path: str

class TextInput(BaseModel):
    text: str

@app.post("/process_documents")
async def process_documents(input: DocumentInput):
    try:
        if not os.path.exists(input.file_path):
            raise HTTPException(status_code=404, detail="File not found")
        loader = TextLoader(input.file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory="/app/chroma_db",
            collection_name="mkdocs_collection"
        )
        vector_store.persist()
        return {"status": "success", "message": f"Processed {len(split_docs)} chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def create_embedding(input: TextInput):
    try:
        embedding = embedding_model.encode(input.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### **3.3.2. `api/requirements.txt`**

fastapi==0.115.0
uvicorn==0.30.6
langchain==0.3.0
langchain-community==0.3.0
chromadb==0.5.5
sentence-transformers==3.1.1

#### **3.3.3. `api/Dockerfile`**

<xaiArtifact artifact_id="82dc107d-750e-4802-9b62-366cf63b0dc8" artifact_version_id="4bbfb92d-72a8-4420-9390-f6db97b576f4" title="Dockerfile" contentType="text/plain">
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
</xaiArtifact>

---

### **3.4. Streamlit 애플리케이션**

Streamlit UI로 사용자 질문을 처리하고 n8n 워크플로우와 통신합니다.

#### **3.4.1. `streamlit/app.py`**

```python
import streamlit as st
import requests
import json

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("MkDocs RAG 챗봇 with n8n")
st.markdown("MkDocs 문서를 기반으로 질문에 답변하는 챗봇입니다. n8n 워크플로우를 통해 처리됩니다.")

N8N_WORKFLOW_URL = "http://n8n:5678/api/v1/workflows/run"

with st.form("chat_form"):
    user_input = st.text_input("질문을 입력하세요:", placeholder="MkDocs에 대해 궁금한 점을 물어보세요!")
    submit_button = st.form_submit_button("전송")

if submit_button and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    try:
        response = requests.post(
            N8N_WORKFLOW_URL,
            json={"question": user_input},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "답변을 생성하지 못했습니다.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.error(f"n8n 워크플로우 호출 실패: {response.status_code}")
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
```

#### **3.4.2. `streamlit/requirements.txt`**

streamlit==1.38.0
requests==2.32.3

#### **3.4.3. `streamlit/Dockerfile`**

<xaiArtifact artifact_id="bade5842-63f0-41df-9765-8c6f2cd854ed" artifact_version_id="942a86da-b6cd-41bf-9665-41042c4728a9" title="Dockerfile" contentType="text/plain">
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
</xaiArtifact>

---

### **3.5. `.gitignore`**

불필요한 파일을 제외합니다.

chroma_db/
**pycache**/
\*.pyc
.env

---

### **3.6. 샘플 MkDocs 문서**

`my-docs/docs/sample.md`를 생성하여 테스트용 문서를 추가합니다.

<xaiArtifact artifact_id="4700b1f5-453f-44f6-9831-75b999dc4014" artifact_version_id="af041036-34e7-48c6-9213-fc606e5d2aa9" title="sample.md" contentType="text/markdown">
# MkDocs 테마 변경

MkDocs에서 테마를 변경하려면 `mkdocs.yml` 파일에서 `theme` 설정을 수정하세요. 예를 들어, Material 테마를 사용하려면:

```yaml
theme:
  name: material
```

사용 가능한 테마는 MkDocs 공식 문서에서 확인할 수 있습니다.
</xaiArtifact>

---

## **4. n8n 워크플로우 설정**

n8n 대시보드(`http://localhost:5678`)에서 워크플로우를 수동으로 설정합니다:

1. **Webhook 노드**:

   - 설정: `HTTP Method: POST`, `Path: /api/v1/workflows/run`
   - Streamlit에서 질문 수신.

2. **HTTP Request 노드 1 (임베딩 생성)**:

   - URL: `http://fastapi:8000/embed`
   - Method: POST
   - Body:
     ```json
     {
       "text": "{{$node['Webhook'].json.body.question}}"
     }
     ```

3. **HTTP Request 노드 2 (ChromaDB 검색)**:

   - URL: `http://chromadb:8001/api/collections/mkdocs_collection/query`
   - Method: POST
   - Body:
     ```json
     {
         "query_embeddings": [{{$node['HTTP Request 1'].json.embedding}}],
         "n_results": 3
     }
     ```

4. **HTTP Request 노드 3 (LM Studio 호출)**:

   - URL: `http://host.docker.internal:1234/v1/completions`
   - Method: POST
   - Body:
     ```json
     {
       "prompt": "다음 문서를 기반으로 질문에 답변하세요:\n{{join($node['HTTP Request 2'].json.documents, '\n\n')}}\n\n질문: {{$node['Webhook'].json.body.question}}\n\n답변:",
       "model": "Mistral-7B-Instruct-v0.1",
       "temperature": 0.7
     }
     ```

5. **Set 노드**:
   - 설정: `answer: {{$node['HTTP Request 3'].json.choices[0].text}}`

---

## **5. 실행 단계**

1. **LM Studio 실행**:

   ```bash
   lmstudio start
   ```

   - 모델: `Mistral-7B-Instruct-v0.1`
   - API: `http://localhost:1234/v1`

2. **디렉토리 생성 및 파일 복사**:

   ```
   rag-project/
   ├── chroma_db/
   ├── my-docs/docs/sample.md
   ├── api/api.py
   ├── api/requirements.txt
   ├── api/Dockerfile
   ├── streamlit/app.py
   ├── streamlit/requirements.txt
   ├── streamlit/Dockerfile
   ├── docker-compose.yml
   ├── .env
   ├── .gitignore
   ```

3. **Docker Compose 실행**:

   ```bash
   cd rag-project
   docker compose up --build
   ```

4. **문서 처리**:

   - FastAPI의 `/process_documents` 호출:
     ```bash
     curl -X POST http://localhost:8000/process_documents -d '{"file_path": "/app/my-docs/docs/sample.md"}' -H "Content-Type: application/json"
     ```

5. **Streamlit 접속**:

   - `http://localhost:8501`에서 질문 입력 (예: "MkDocs에서 테마를 변경하는 방법은?").

6. **n8n 워크플로우 활성화**:
   - n8n 대시보드에서 워크플로우를 활성화.

---

## **6. 테스트 및 디버깅**

- **문서 처리**:
  - `./chroma_db`에 SQLite 파일과 인덱스 생성 확인.
- **질문 테스트**:
  - Streamlit에서 질문 입력, 예상 답변: "MkDocs.yml에서 `theme` 설정을 변경하세요."
- **디버깅**:
  - ChromaDB 로그: `docker logs rag-project-chromadb-1`
  - n8n 워크플로우 실행 기록 확인.
  - FastAPI 로그: `docker logs rag-project-fastapi-1`

---

## **7. 추가 고려사항**

- **권한**: `./chroma_db`와 `./my-docs/docs`에 `chmod -R 777` 설정.
- **확장성**: n8n 워크플로우에 PDF 로더 추가 가능.
- **보안**: FastAPI와 n8n에 API 키 인증 추가.
- **백업**: `./chroma_db`를 주기적으로 백업.

---

## **8. 결론**

단일 `docker-compose.yml`으로 n8n, ChromaDB, FastAPI, Streamlit을 통합하고, `./chroma_db`와 MkDocs 문서를 볼륨으로 공유했습니다. LM Studio는 호스트에서 실행되며, 네트워크를 통해 통합됩니다. 위 코드를 따라 설정하면 완전한 RAG 시스템이 동작합니다.
