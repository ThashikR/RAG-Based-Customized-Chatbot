# RAG-Based-Customized-Chatbot

# PDF Question & Answer System

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content. The system uses LangChain, Ollama, and HuggingFace embeddings to provide accurate answers based solely on the document's content.

## Features

- PDF document upload and processing
- Interactive question-answering interface
- Source page references for answers
- Real-time processing status updates
- User-friendly interface

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- The llama3.1 model downloaded in Ollama

### Installing Ollama and Models

1. Install Ollama by following instructions at [Ollama's official website](https://ollama.ai/)
2. Pull the required model:
```bash
ollama pull llama3.1
