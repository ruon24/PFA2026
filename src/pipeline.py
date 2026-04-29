"""
Main Pipeline - Orchestrates the complete RAG pipeline
"""

import os
import glob
from typing import List, Dict, Optional
import time

from src.pdf_parser import PDFParser
from src.chunker import TextChunker
from src.embedder import EmbeddingGenerator
from src.vector_store import VectorStore
from src.query_engine import QueryEngine


class Pipeline:
    """Main RAG Pipeline orchestrator"""
    
    def __init__(
        self,
        collection_name: str = "pdf_rag",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3.2"
    ):
        """
        Initialize the pipeline
        
        Args:
            collection_name: Name for the ChromaDB collection
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlapping tokens between chunks
            embedding_model: Sentence-transformers model name
            ollama_model: Ollama model name
        """
        self.collection_name = collection_name
        
        # Initialize components
        self.pdf_parser = PDFParser()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.vector_store = VectorStore(collection_name=collection_name)
        self.query_engine = QueryEngine(model=ollama_model)
        
        print(f"Pipeline initialized with collection: {collection_name}")
    
    def ingest_pdfs(self, pdf_folder: str, verbose: bool = True) -> Dict:
        """
        Ingest all PDFs from a folder
        
        Args:
            pdf_folder: Path to folder containing PDFs
            verbose: Print progress information
            
        Returns:
            Dictionary with ingestion statistics
        """
        pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
        
        if not pdf_files:
            return {"status": "no_files", "message": f"No PDF files found in {pdf_folder}"}
        
        stats = {
            "total_files": len(pdf_files),
            "total_chunks": 0,
            "files_processed": 0
        }
        
        for idx, pdf_path in enumerate(pdf_files):
            if verbose:
                print(f"\n[{idx+1}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
            
            try:
                # Extract text from PDF
                text = self.pdf_parser.extract_text(pdf_path)
                
                if verbose:
                    print(f"  Extracted {len(text)} characters")
                
                # Chunk text
                chunks = self.chunker.chunk_text(text)
                stats["total_chunks"] += len(chunks)
                
                if verbose:
                    print(f"  Created {len(chunks)} chunks")
                
                # Generate embeddings
                embeddings = self.embedder.generate_embeddings(chunks)
                
                if verbose:
                    print(f"  Generated {len(embeddings)} embeddings")
                
                # Store in ChromaDB
                file_id = os.path.splitext(os.path.basename(pdf_path))[0]
                ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [
                    {"source": os.path.basename(pdf_path), "chunk_id": i, "file_id": file_id}
                    for i in range(len(chunks))
                ]
                
                self.vector_store.add_documents(ids, chunks, embeddings, metadatas)
                
                if verbose:
                    print(f"  Stored in ChromaDB")
                
                stats["files_processed"] += 1
                
            except Exception as e:
                if verbose:
                    print(f"  Error: {str(e)}")
                continue
        
        if verbose:
            print(f"\n✓ Ingestion complete: {stats['files_processed']}/{stats['total_files']} files, {stats['total_chunks']} chunks")
        
        return stats
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Query the RAG pipeline
        
        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        # Check Ollama connection
        if not self.query_engine.check_connection():
            return {
                "status": "error",
                "message": "Ollama is not running. Start with: ollama serve"
            }
        
        # Generate embedding for question
        query_embedding = self.embedder.generate_embedding(question)
        
        # Search ChromaDB
        results = self.vector_store.query(query_embedding, n_results=top_k)
        
        # Get context from results
        if not results['documents'] or not results['documents'][0]:
            return {
                "status": "no_results",
                "message": "No relevant documents found. Try ingesting PDFs first."
            }
        
        context = "\n\n".join(results['documents'][0])
        
        # Get answer from Ollama
        answer = self.query_engine.generate(question, context)
        
        # Get sources
        sources = []
        if results['metadatas'] and results['metadatas'][0]:
            for meta in results['metadatas'][0]:
                sources.append(meta.get('source', 'Unknown'))
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "sources": list(set(sources)),
            "num_chunks_retrieved": len(results['documents'][0]) if results['documents'] else 0
        }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        collection_info = self.vector_store.get_collection_info()
        return {
            "collection_name": collection_info["name"],
            "document_count": collection_info["count"],
            "embedding_dimension": self.embedder.get_embedding_dimension(),
            "ollama_connected": self.query_engine.check_connection()
        }
    
    def reset(self):
        """Reset the pipeline by deleting the collection"""
        self.vector_store.delete_collection()
        print(f"Collection '{self.collection_name}' deleted. Reinitializing...")
        self.vector_store = VectorStore(collection_name=self.collection_name)


def main():
    """Main entry point for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PFA26 RAG Pipeline")
    parser.add_argument("--pdfs", default="./data", help="PDF folder path")
    parser.add_argument("--query", help="Query string")
    parser.add_argument("--stats", action="store_true", help="Show pipeline stats")
    parser.add_argument("--reset", action="store_true", help="Reset the pipeline")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    if args.reset:
        pipeline.reset()
    
    if args.stats:
        stats = pipeline.get_stats()
        print("\n=== Pipeline Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    if args.pdfs:
        # Ingest PDFs
        print(f"\nIngesting PDFs from: {args.pdfs}")
        pipeline.ingest_pdfs(args.pdfs)
    
    if args.query:
        # Query
        print(f"\nQuery: {args.query}")
        result = pipeline.query(args.query)
        
        print("\n=== Answer ===")
        print(result.get("answer", result.get("message", "No answer")))
        
        if result.get("sources"):
            print(f"\nSources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()