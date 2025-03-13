from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import re
import html2text
from tqdm import tqdm
import numpy as np
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.docstore.document import Document as LangchainDocument
from qdrant_client import QdrantClient
from qdrant_client.http import models

class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_url: str = "http://localhost:6333",
        qdrant_collection_name: str = "default_collection",
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize the RAG Pipeline with configurable parameters.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            embedding_model: Name of the HuggingFace model to use for embeddings
            qdrant_url: URL for Qdrant server or Qdrant Cloud cluster
            qdrant_collection_name: Name of the collection in Qdrant
            qdrant_api_key: API key for Qdrant Cloud authentication
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.qdrant_url = qdrant_url
        self.qdrant_collection_name = qdrant_collection_name
        self.qdrant_api_key = qdrant_api_key
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Qdrant client with API key if provided
        client_kwargs = {
            "url": qdrant_url,
            "timeout": 100,  # Increased timeout
            "prefer_grpc": False  # Use HTTP
        }
        if qdrant_api_key:
            client_kwargs["api_key"] = qdrant_api_key
            
        self.qdrant_client = QdrantClient(**client_kwargs)
        self.vector_store = None
        
        # Initialize vector store
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """
        Initialize the Qdrant vector store and create collection if it doesn't exist.
        """
        try:
            # Get embedding dimension
            sample_embedding = self.embedding_model.embed_query("sample text")
            embedding_dim = len(sample_embedding)
            
            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections().collections
            exists = any(col.name == self.qdrant_collection_name for col in collections)
            
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.qdrant_collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created new Qdrant collection: {self.qdrant_collection_name}")
            
            # Initialize vector store
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.qdrant_collection_name,
                embeddings=self.embedding_model
            )
            
        except Exception as e:
            print(f"Error initializing Qdrant: {str(e)}")
            raise

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[LangchainDocument]:
        """
        Process a list of documents by splitting them into chunks.
        
        Args:
            documents: List of dictionaries containing 'text' and 'metadata' keys
            
        Returns:
            List of LangchainDocument objects with processed chunks
        """
        processed_documents = []
        for doc in documents:
            # Create Langchain Document objects
            langchain_doc = LangchainDocument(
                page_content=doc['text'],
                metadata=doc['metadata']
            )
            # Split the document into chunks
            chunks = self.text_splitter.split_documents([langchain_doc])
            processed_documents.extend(chunks)
        
        print(f"Split {len(documents)} documents into {len(processed_documents)} chunks")
        return processed_documents

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embedding_model.embed_documents(texts)

    def create_vector_store(self, documents: List[LangchainDocument]) -> None:
        """
        Create a vector store from processed documents.
        
        Args:
            documents: List of Document objects to embed and store
        """
        try:
            print("Adding documents to Qdrant vector store...")
            # Process in smaller batches
            batch_size = 50  # Smaller batch size
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1} of {(len(documents)-1)//batch_size + 1}")
                
                max_retries = 3
                retry_delay = 5  # seconds
                
                for attempt in range(max_retries):
                    try:
                        self.vector_store.add_documents(batch)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:  # Last attempt
                            raise
                        print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                
                # Small delay between batches
                time.sleep(1)
                
            print(f"Added {len(documents)} documents to Qdrant collection '{self.qdrant_collection_name}'")
        except Exception as e:
            print(f"Error adding documents to Qdrant: {str(e)}")
            raise

    def retrieve_relevant_chunks(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve the k most relevant chunks for a given query.
        
        Args:
            query: The search query
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized.")
            
        return self.vector_store.similarity_search(query, k=k)

    def delete_collection(self) -> None:
        """
        Delete the current Qdrant collection.
        """
        try:
            self.qdrant_client.delete_collection(self.qdrant_collection_name)
            print(f"Deleted Qdrant collection: {self.qdrant_collection_name}")
            self.vector_store = None
        except Exception as e:
            print(f"Error deleting Qdrant collection: {str(e)}")
            raise

    def get_collection_info(self) -> Dict:
        """
        Get information about the current Qdrant collection.
        
        Returns:
            Dictionary containing collection information
        """
        try:
            info = self.qdrant_client.get_collection(self.qdrant_collection_name)
            return {
                "collection_name": self.qdrant_collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status)
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            raise