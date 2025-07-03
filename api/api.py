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