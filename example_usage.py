from rag_pipeline import RAGPipeline
from web_scraper import WebScraper
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Qdrant Cloud credentials from environment variables
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Validate environment variables
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError(
            "Please set QDRANT_URL and QDRANT_API_KEY environment variables. "
            "You can find these in your Qdrant Cloud dashboard."
        )
    
    if not HUGGINGFACE_TOKEN:
        raise ValueError(
            "Please set HUGGINGFACEHUB_API_TOKEN environment variable. "
            "You can get it from https://huggingface.co/settings/tokens"
        )
    
    try:
        # Initialize the web scraper and get documents
        print("\nScraping Dr. Berg's blog content...")
        scraper = WebScraper(
            base_url="https://www.drberg.com/blog",
            max_pages=5  # Scrape more pages to get a good selection
        )
        documents = scraper.scrape_website()
        
        if not documents:
            print("Trying articles section...")
            scraper = WebScraper(
                base_url="https://www.drberg.com/blog",
                max_pages=5
            )
            documents = scraper.scrape_website()
        
        if not documents:
            raise ValueError("No documents were scraped from the website")
        
        print(f"Successfully scraped {len(documents)} pages")
        
        # Initialize the RAG pipeline
        pipeline = RAGPipeline(
            chunk_size=500,
            chunk_overlap=50,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            qdrant_url=QDRANT_URL,
            qdrant_collection_name="drberg_blogs",
            qdrant_api_key=QDRANT_API_KEY
        )
        
        # Process documents and create chunks
        print("\nProcessing documents...")
        processed_chunks = pipeline.process_documents(documents)
        
        # Show details of scraped documents
        print("\nScraped Documents Overview:")
        print(f"Total documents: {len(documents)}")
        print("\nFirst 3 documents:")
        for i, doc in enumerate(documents[:3]):
            print(f"\nDocument {i+1}:")
            print(f"Title: {doc['metadata']['title']}")
            print(f"Source: {doc['metadata']['source']}")
            print(f"Content preview: {doc['text'][:200]}...")
            print("-" * 80)
            
        # Show details of processed chunks
        print("\nProcessed Chunks Overview:")
        print(f"Total chunks: {len(processed_chunks)}")
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(processed_chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Content length: {len(chunk.page_content)} characters")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content preview: {chunk.page_content[:200]}...")
            print("-" * 80)
        
        # Add documents to vector store
        print("\nAdding documents to vector store...")
        pipeline.create_vector_store(processed_chunks)
        
        # Show vector store information
        collection_info = pipeline.get_collection_info()
        print("\nVector Store Information:")
        print(f"Collection name: {collection_info['collection_name']}")
        print(f"Number of vectors: {collection_info['vectors_count']}")
        print(f"Number of points: {collection_info['points_count']}")
        print(f"Status: {collection_info['status']}")
        print("-" * 80)
        
        # Example queries to find the best content
        queries = [
            "What are The Best Nutrients for Healthy Adrenals?",
        ]
        
        print("\nSearching for the best articles...")
        llm = HuggingFaceHub(
            repo_id="facebook/bart-large-cnn",
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512,
                "do_sample": True
            }
        )
        
        # 2. Setup retriever with more chunks
        retriever = pipeline.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 3. Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True  # This will return the chunks used for generation
        )
        
        # Process queries
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Question: {query}")
            print(f"{'='*80}")
            
            # Get answer with source documents
            result = qa_chain.invoke({"query": query})
            
            # Show retrieved chunks
            print("\nRetrieved Relevant Chunks:")
            print("-" * 80)
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\nChunk {i}:")
                print(f"Source: {doc.metadata['source']}")
                print(f"Content: {doc.page_content[:200]}...")
                print("-" * 40)
            
            # Show generated answer
            print("\nGenerated Answer:")
            print("-" * 80)
            print(result["result"])
            print("\n" + "="*80)
        
        # Clean up
        # pipeline.delete_collection()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 