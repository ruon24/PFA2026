"""
RAG Pipeline: PDF → Parse → Chunk → Embed → ChromaDB → Query → Ollama
"""

import os
import glob
from typing import List, Dict
from pypdf import PdfReader
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ollama
import tiktoken

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
COLLECTION_NAME = "pdf_rag"


class PDFProcessor:
    """Parse PDFs and extract text"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from a PDF file using pdfplumber"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text


class TextChunker:
    """Split text into chunks"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.enc.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.enc.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts, show_progress_bar=True).tolist()


class VectorStore:
    """Store and query embeddings in ChromaDB"""
    
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """Add documents to the vector store"""
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def query(self, query_embedding: List[float], n_results: int = 3) -> Dict:
        """Query the vector store"""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )


class OllamaClient:
    """Query Ollama for answers"""
    
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
    
    def generate(self, prompt: str, context: str) -> str:
        """Generate a response using Ollama"""
        full_prompt = f"""Based on the following context, answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {prompt}

Answer:"""
        
        response = ollama.generate(
            model=self.model,
            prompt=full_prompt
        )
        return response['response']


class RAGPipeline:
    """Main RAG Pipeline"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.ollama = OllamaClient()
    
    def ingest_pdfs(self, pdf_folder: str):
        """Ingest PDFs from a folder"""
        pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
        
        for idx, pdf_path in enumerate(pdf_files):
            print(f"Processing: {pdf_path}")
            
            # Extract text
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Chunk text
            chunks = self.chunker.chunk_text(text)
            print(f"  Created {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = self.embedder.generate_embeddings(chunks)
            
            # Store in ChromaDB
            ids = [f"doc_{idx}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": os.path.basename(pdf_path), "chunk_id": i} for i in range(len(chunks))]
            
            self.vector_store.add_documents(ids, chunks, embeddings, metadatas)
            print(f"  Stored in ChromaDB")
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG pipeline"""
        # Generate embedding for question
        query_embedding = self.embedder.generate_embeddings([question])[0]
        
        # Search ChromaDB
        results = self.vector_store.query(query_embedding, top_k)
        
        # Get context
        context = "\n\n".join(results['documents'][0])
        
        # Get answer from Ollama
        answer = self.ollama.generate(question, context)
        
        return answer


def main():
    """Example usage"""
    pipeline = RAGPipeline()
    
    # Ingest PDFs (uncomment and modify the path)
    pipeline.ingest_pdfs("./pdfs")
    
    # Query (uncomment to test)
    answer = pipeline.query("What is the main topic of the document?")
    print(answer)


if __name__ == "__main__":
    main()