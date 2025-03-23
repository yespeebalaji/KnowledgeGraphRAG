import os
import sys

# Configure environment before any other imports
os.environ['TORCH_USE_RTLD_GLOBAL'] = 'YES'
os.environ['STREAMLIT_TORCH_DEBUG'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Prevent torch path examination
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.pathsep + os.environ['PYTHONPATH']
else:
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Patch asyncio before Streamlit import
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if not asyncio._get_running_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Pre-initialize torch to prevent path examination
try:
    import torch
    def _mock_path_func(*args, **kwargs):
        return []
    if hasattr(torch, '_classes'):
        torch._classes.__path__ = type('_MockPath', (), {'_path': property(_mock_path_func)})
        torch.classes.__path__ = type('_MockPath', (), {'_path': property(_mock_path_func)})
except ImportError:
    pass

# Now import other modules
import streamlit as st
import logging
from typing import Optional
import io
import requests
from pathlib import Path
from datetime import datetime
import atexit

import importlib.util
import sys

def import_local_module(module_name: str):
    """Import a module from the local app directory."""
    module_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import local modules
config = import_local_module("config")
DocumentProcessor = import_local_module("document_processor").DocumentProcessor
RAGEngine = import_local_module("rag_engine").RAGEngine

# Extract config variables
STREAMLIT_TITLE = config.STREAMLIT_TITLE
STREAMLIT_DESCRIPTION = config.STREAMLIT_DESCRIPTION
ALLOWED_FILE_TYPES = config.ALLOWED_FILE_TYPES
MAX_FILE_SIZE = config.MAX_FILE_SIZE

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def cleanup():
    """Clean up resources."""
    if 'rag_components' in st.session_state and not st.session_state.get('cleanup_done', False):
        logger.info("Starting resource cleanup")
        try:
            logger.info("Closing RAG engine")
            st.session_state['rag_components']['rag_engine'].close()
            logger.info("RAG components cleaned up successfully")
            st.session_state.cleanup_done = True
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)

# Register cleanup function to run at exit
atexit.register(cleanup)

def initialize_components():
    """Initialize RAG system components."""
    logger.info("Starting component initialization")
    if 'rag_components' not in st.session_state:
        try:
            logger.info("Creating DocumentProcessor instance")
            doc_processor = DocumentProcessor()
            
            logger.info("Creating RAGEngine instance")
            rag_engine = RAGEngine()
            
            logger.info("Storing components in session state")
            st.session_state['rag_components'] = {
                'doc_processor': doc_processor,
                'rag_engine': rag_engine
            }
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}", exc_info=True)
            st.error(f"Error initializing components: {str(e)}")
            raise
    else:
        logger.info("Using existing components from session state")
    
    return (
        st.session_state['rag_components']['doc_processor'],
        st.session_state['rag_components']['rag_engine']
    )

def init_session_state():
    """Initialize session state variables."""
    logger.info("Initializing session state")
    if 'chat_history' not in st.session_state:
        logger.debug("Creating chat history in session state")
        st.session_state.chat_history = []
    if 'documents' not in st.session_state:
        logger.debug("Creating documents list in session state")
        st.session_state.documents = []

def render_chat_history():
    """Render chat message history."""
    logger.debug(f"Rendering chat history with {len(st.session_state.chat_history)} messages")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def add_message(role: str, content: str):
    """Add a message to the chat history."""
    logger.debug(f"Adding message with role '{role}' to chat history")
    st.session_state.chat_history.append({"role": role, "content": content})

def process_file_upload(doc_processor: DocumentProcessor, 
                       rag_engine: RAGEngine,
                       uploaded_file) -> Optional[str]:
    """Process an uploaded file."""
    logger.info(f"Processing uploaded file: {uploaded_file.name}")
    try:
        # Validate file
        if uploaded_file.name.split('.')[-1].lower() not in ALLOWED_FILE_TYPES:
            logger.warning(f"Unsupported file type: {uploaded_file.name}")
            st.error(f"Unsupported file type. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}")
            return None
        
        if uploaded_file.size > MAX_FILE_SIZE * 1024 * 1024:
            logger.warning(f"File size exceeds limit: {uploaded_file.size} bytes")
            st.error(f"File size exceeds {MAX_FILE_SIZE}MB limit")
            return None
        
        # Process file content
        logger.info("Reading file content")
        file_content = uploaded_file.read()
        if uploaded_file.type == "application/pdf":
            logger.info("Processing PDF document")
            processed_doc = doc_processor.process_document(
                io.BytesIO(file_content),
                'pdf'
            )
        else:  # text file
            logger.info("Processing text document")
            processed_doc = doc_processor.process_document(
                file_content.decode(),
                'text'
            )
        
        # Add to RAG system
        logger.info("Adding document to RAG system")
        doc_id = rag_engine.add_document(
            content=processed_doc['original_text'],
            doc_type=uploaded_file.type,
            metadata={
                'filename': uploaded_file.name,
                'upload_time': datetime.now().isoformat(),
                'file_type': uploaded_file.type,
                **processed_doc['metadata']
            }
        )
        
        logger.info(f"Document processed successfully with ID: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        st.error(f"Error processing file: {str(e)}")
        return None

def process_url(doc_processor: DocumentProcessor,
                rag_engine: RAGEngine,
                url: str) -> Optional[str]:
    """Process a URL."""
    logger.info(f"Processing URL: {url}")
    try:
        # Validate URL
        logger.info("Validating URL")
        response = requests.head(url)
        if response.status_code != 200:
            logger.warning(f"Invalid URL or inaccessible content: {url}")
            st.error("Invalid URL or content not accessible")
            return None
        
        # Process URL content
        logger.info("Processing URL content")
        processed_doc = doc_processor.process_document(url, 'url')
        
        # Add to RAG system
        logger.info("Adding URL content to RAG system")
        doc_id = rag_engine.add_document(
            content=processed_doc['original_text'],
            doc_type='url',
            metadata={
                'url': url,
                'upload_time': datetime.now().isoformat(),
                **processed_doc['metadata']
            }
        )
        
        logger.info(f"URL processed successfully with ID: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}", exc_info=True)
        st.error(f"Error processing URL: {str(e)}")
        return None

def main():
    """Main Streamlit application."""
    logger.info("Starting application")

    # Set up Streamlit page configuration
    st.set_page_config(
        page_title=STREAMLIT_TITLE,
        page_icon="📚",
        layout="wide"
    )
    
    try:
        # Initialize components
        doc_processor, rag_engine = initialize_components()
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize RAG components: {str(e)}")
        return
    
    # Initialize session state
    init_session_state()
    
    # Title and description
    st.title(STREAMLIT_TITLE)
    st.markdown(STREAMLIT_DESCRIPTION)
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Add Documents")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=ALLOWED_FILE_TYPES,
            help=f"Maximum file size: {MAX_FILE_SIZE}MB"
        )
        if uploaded_file:
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    doc_id = process_file_upload(doc_processor, rag_engine, uploaded_file)
                    if doc_id:
                        st.success("File processed successfully!")
                        st.session_state.documents.append({
                            'id': doc_id,
                            'name': uploaded_file.name,
                            'type': 'file'
                        })
        
        # URL input
        st.markdown("---")
        url = st.text_input("Or enter a URL")
        if url:
            if st.button("Process URL"):
                with st.spinner("Processing URL..."):
                    doc_id = process_url(doc_processor, rag_engine, url)
                    if doc_id:
                        st.success("URL processed successfully!")
                        st.session_state.documents.append({
                            'id': doc_id,
                            'name': url,
                            'type': 'url'
                        })
        
        # Document list
        if st.session_state.documents:
            st.markdown("---")
            st.header("📚 Processed Documents")
            for doc in st.session_state.documents:
                st.text(f"• {doc['name']}")
    
    # Main chat interface
    st.markdown("---")
    render_chat_history()
    
    # Query input
    if query := st.chat_input("Ask a question about your documents"):
        logger.info("Processing user query")
        # Add user message to chat
        add_message("user", query)
        
        # Process query
        with st.spinner("Thinking..."):
            try:
                logger.info("Sending query to RAG engine")
                result = rag_engine.process_query(query)
                logger.info("Query processed successfully")
                add_message("assistant", result['response'])
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                add_message("assistant", "I apologize, but I encountered an error processing your query.")

if __name__ == "__main__":
    main()
