from src.pipeline import Pipeline


def main():
    """Run the RAG pipeline"""
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Ingest PDFs from data folder
    print("\n=== Ingesting PDFs ===")
    stats = pipeline.ingest_pdfs("./data")
    
    print(f"\n=== Ingestion Complete ===")
    if stats.get('status') == 'no_files':
        print(f"No PDF files found in ./data folder")
        print("Place PDF files in the data/ folder and run again")
    else:
        print(f"Files processed: {stats.get('files_processed', 0)}/{stats.get('total_files', 0)}")
        print(f"Total chunks: {stats.get('total_chunks', 0)}")
    
    # Example queries
    print("\n=== Example Queries ===")
    
    # Check if there are any documents in the vector store
    collection_info = pipeline.vector_store.get_collection_info()
    if collection_info['count'] == 0:
        print("No documents in vector store. Skipping queries.")
    else:
        queries = [
            "What is the main topic of the documents?",
            "Summarize the key points"
        ]
        
        for query in queries:
            print(f"\nQ: {query}")
            result = pipeline.query(query)
            
            if result['status'] == 'success':
                print(f"A: {result['answer']}")
                print(f"Sources: {', '.join(result['sources'])}")
            else:
                print(f"Error: {result['message']}")


if __name__ == "__main__":
    main()