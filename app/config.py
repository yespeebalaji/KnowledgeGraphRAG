import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'RAG')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'Rag@2025')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

# Google API Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformers model
VECTOR_SIMILARITY_THRESHOLD = 0.5  # Lower threshold for better recall

# RAG Configuration
MAX_CONTEXT_LENGTH = 4000  # Maximum context length for LLM
TOP_K_RESULTS = 5  # Number of relevant chunks to retrieve

# Streamlit Configuration
STREAMLIT_TITLE = " Knowledge Graph RAG"
STREAMLIT_DESCRIPTION = """
A Retrieval-Augmented Generation (RAG) system using Neo4j knowledge graphs 
and LLMs to query  data with enhanced accuracy.
"""

# File Upload Configuration
ALLOWED_FILE_TYPES = ["pdf", "txt"]
MAX_FILE_SIZE = 10  # in MB
