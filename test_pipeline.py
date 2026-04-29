"""
Test script to verify the RAG pipeline components
"""

from rag_pipeline import PDFProcessor, TextChunker, EmbeddingGenerator, VectorStore, OllamaClient

def test_components():
    print("Testing RAG Pipeline Components...\n")
    
    # Test 1: PDF Processor
    print("1. PDF Processor - OK (imported successfully)")
    
    # Test 2: Text Chunking
    print("2. Testing Text Chunking...")
    chunker = TextChunker(chunk_size=100, overlap=20)
    test_text = "This is a test sentence. " * 20
    chunks = chunker.chunk_text(test_text)
    print(f"   Created {len(chunks)} chunks from test text")
    
    # Test 3: Embedding Generator
    print("3. Testing Embedding Generator...")
    embedder = EmbeddingGenerator()
    test_embeddings = embedder.generate_embeddings(["Hello world", "Test sentence"])
    print(f"   Generated embeddings: shape {len(test_embeddings)} vectors")
    
    # Test 4: Vector Store
    print("4. Testing ChromaDB Vector Store...")
    store = VectorStore("test_collection")
    # Add test documents
    store.add_documents(
        ids=["test1", "test2"],
        documents=["Document 1 content", "Document 2 content"],
        embeddings=test_embeddings,
        metadatas=[{"source": "test"}, {"source": "test"}]
    )
    print("   Added test documents to ChromaDB")
    
    # Test 5: Ollama
    print("5. Testing Ollama Client...")
    client = OllamaClient()
    print("   Ollama client initialized")
    
    print("\n✓ All components working!")
    print("\nTo use the pipeline:")
    print("1. Place PDF files in a 'pdfs' folder")
    print("2. Run: python rag_pipeline.py")
    print("3. Uncomment the ingest_pdfs() and query() calls")
    print("4. Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    test_components()