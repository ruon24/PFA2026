"""
Vector Store - ChromaDB storage and retrieval
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os


class VectorStore:
    """Store and query embeddings in ChromaDB"""
    
    def __init__(self, collection_name: str = "pdf_rag", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ):
        """
        Add documents to the vector store
        
        Args:
            ids: Unique identifiers for documents
            documents: Text documents
            embeddings: Embedding vectors
            metadatas: Optional metadata for each document
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 3,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Query the vector store
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Filter by metadata
            where_document: Filter by document content
            
        Returns:
            Query results with documents, distances, and metadatas
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
    
    def get_by_id(self, ids: List[str]) -> Dict:
        """
        Get documents by IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            Documents with their data
        """
        return self.collection.get(ids=ids)
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory
        }
    
    def peek(self, limit: int = 10) -> Dict:
        """Peek at the first N documents in the collection"""
        return self.collection.peek(limit=limit)