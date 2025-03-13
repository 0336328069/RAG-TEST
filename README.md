# RAG Pipeline with LangChain

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain. The pipeline includes data cleaning, text chunking, embedding generation, vector storage, and retrieval components.

## Features

- Data cleaning with HTML tag removal and text normalization
- Intelligent text chunking with overlapping tokens
- Embedding generation using Hugging Face's sentence-transformers
- Vector storage using FAISS
- Retrieval mechanism with similarity search
- Integration with language models for answer generation

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root and add your HuggingFace API token:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Usage

The pipeline is implemented in the `RAGPipeline` class. Here's a basic example of how to use it:

```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(
    chunk_size=500,
    chunk_overlap=50
)

# Process documents
documents = [
    {
        'text': 'Your text here...',
        'metadata': {'source': 'example', 'section': 'intro'}
    }
]
chunks = pipeline.process_documents(documents)

# Create vector store
pipeline.create_vector_store(chunks)

# Retrieve relevant chunks for a query
results = pipeline.retrieve_relevant_chunks("Your query here", k=3)

# Save vector store for later use
pipeline.save_vector_store("vector_store")
```

For a complete example, see `example_usage.py`.

## Pipeline Components

1. **Data Cleaning**
   - Removes HTML tags
   - Normalizes whitespace
   - Removes special characters

2. **Text Chunking**
   - Uses RecursiveCharacterTextSplitter
   - Configurable chunk size and overlap
   - Preserves semantic coherence

3. **Embedding Generation**
   - Uses sentence-transformers/all-MiniLM-L6-v2
   - Configurable model selection

4. **Vector Storage**
   - FAISS implementation
   - Efficient similarity search
   - Persistent storage support

## Configuration

The pipeline can be configured with the following parameters:

- `chunk_size`: Size of text chunks (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `embedding_model`: HuggingFace model name for embeddings

## License

MIT License 