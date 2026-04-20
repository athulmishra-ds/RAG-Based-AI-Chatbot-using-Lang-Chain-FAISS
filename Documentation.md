import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import uvicorn

app = FastAPI(title="RAG Chatbot API")

# Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db_path = "faiss_index"
qa_chain = None

def build_qa_chain():
    """Initializes the RAG chain using the local FAISS index."""
    global qa_chain
    if os.path.exists(vector_db_path):
        vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        
        # Integrating Hugging Face Transformers for answering
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": 150}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

@app.on_event("startup")
async def startup_event():
    build_qa_chain()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """API to upload and index documents (PDF/Text)."""
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Load and Split documents
    loader = PyPDFLoader(temp_path) if file.filename.endswith('.pdf') else TextLoader(temp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Update or create FAISS vector store
    if os.path.exists(vector_db_path):
        vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        vector_db.add_documents(texts)
    else:
        vector_db = FAISS.from_documents(texts, embeddings)
    
    vector_db.save_local(vector_db_path)
    build_qa_chain() # Refresh chain with new data
    
    os.remove(temp_path)
    return {"status": "Success", "message": f"Indexed {file.filename}"}

@app.post("/chat")
async def chat(query: str = Form(...)):
    """API for chatbot interaction."""
    if qa_chain is None:
        return {"error": "No documents indexed. Please upload a file first."}
    
    response = qa_chain.invoke({"query": query})
    return {
        "answer": response["result"],
        "context_sources": [doc.metadata for doc in response["source_documents"]]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
