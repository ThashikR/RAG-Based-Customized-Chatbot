import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="PDF Q&A System", layout="wide")

# Prompt to restrict the answer to come only from the given document, here Pdf but not limited to
prompt_template = """
You are a helpful assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer from the context provided, just say that you don't have enough information from the document.
Do not try to make up an answer. Your answer should be concise and based ONLY on the provided text.

Context:
{context}

Question:
{question}

Answer:
"""

@st.cache_resource
def create_qa_chain(pdf_file):
    """
    Creates QA chain from uploaded PDF file
    """
    try:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())
        
        # Initialize Ollama
        llm = Ollama(model="llama3.1")

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(pages)

        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

        # Create prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        # Clean up
        os.remove("temp.pdf")
        
        return qa_chain

    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

def main():
    # Add title and description
    st.title("📄 PDF Question & Answer System")
    st.markdown("""
    Upload a PDF document and ask questions about its content.
    The system will provide answers based solely on the information contained in the document.
    """)

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    # File upload section
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    st.session_state.qa_chain = create_qa_chain(uploaded_file)
                    st.success("Document processed successfully!")

    # Question and Answer section
    if st.session_state.qa_chain is not None:
        st.header("Ask Questions")
        question = st.text_input("Enter your question about the document:")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner("Thinking..."):
                    try:
                        # Get response
                        response = st.session_state.qa_chain.invoke({"query": question})
                        
                        # Display answer
                        st.markdown("### Answer:")
                        st.write(response["result"].strip())
                        
                        # Display source pages
                        source_pages = sorted(list(set([doc.metadata.get('page', 'N/A') for doc in response["source_documents"]])))
                        st.markdown("### Source Pages:")
                        st.write(f"Information was retrieved from page(s): {source_pages}")
                        
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please upload a PDF document to get started.")

if __name__ == "__main__":
    main()