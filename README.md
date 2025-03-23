# Historical Knowledge Graph RAG System

A Retrieval-Augmented Generation (RAG) system using Neo4j knowledge graphs and Google's Gemini Pro LLM to query historical data. The system combines graph-based and vector-based retrieval methods for enhanced accuracy.

## Features

- 📚 Multi-format document processing (PDF, URL, Text)
- 🔍 Hybrid retrieval combining vector similarity and graph relationships
- 🧠 Gemini Pro LLM integration for natural language responses
- 📊 Neo4j knowledge graph for structured data representation
- 🌐 Streamlit web interface for easy interaction
- 💾 Document management and chat history

## Prerequisites

- Python 3.8+
- Neo4j Database Server 5.x
- Google Cloud API Key with Gemini Pro access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-rag
```

2. Quick Setup:

**Windows:**
```bash
# Run the setup script
setup.bat
```

**Unix/Linux/MacOS:**
```bash
# Make setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

3. Manual Setup (if scripts don't work):

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install package
pip install -e .
```

**Unix/Linux/MacOS:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install package
pip install -e .
```

4. Set up Neo4j:
- Install and start Neo4j Database
- Create a new database or use the default
- Create a user with appropriate permissions
- Note down the connection details

5. Configure environment variables:
- Copy `.env.example` to `.env`
- Update the following variables:
  ```
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USERNAME=your_username
  NEO4J_PASSWORD=your_password
  NEO4J_DATABASE=neo4j
  GOOGLE_API_KEY=your_google_api_key
  ```

## Usage

1. Start the application:

From the project root directory:
```bash
# Method 1 (Recommended)
python -m streamlit run app/main.py

# Method 2 (Alternative)
streamlit run app/main.py
```

Note: If you encounter import errors, ensure you're running the command from the project root directory and the package is installed correctly (`pip install -e .`).

2. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

3. Using the application:
   - Upload documents through the sidebar
   - Enter URLs to process web content
   - Ask questions in the chat interface
   - View processed documents in the sidebar

## Project Structure

```
pdf_rag/
├── app/
│   ├── __init__.py
│   ├── main.py               # Streamlit interface
│   ├── document_processor.py # Document handling
│   ├── knowledge_graph.py    # Neo4j operations
│   ├── embeddings.py        # Vector operations
│   ├── rag_engine.py        # Query processing
│   └── config.py            # Configuration
├── requirements.txt
└── README.md
```

## Configuration

The system can be configured through:
- Environment variables in `.env`
- Application settings in `app/config.py`

Key configuration options:
- Document processing: chunk size, overlap
- Vector similarity threshold
- Maximum context length for LLM
- File upload limitations

## Development

### Adding New Features

1. Document Processing:
   - Extend `DocumentProcessor` for new file types
   - Implement custom chunking strategies

2. Knowledge Graph:
   - Add new node/relationship types
   - Implement additional graph queries

3. RAG Engine:
   - Enhance entity extraction
   - Modify prompt engineering
   - Add new retrieval strategies

### Testing

- Test document processing with various formats
- Verify knowledge graph operations
- Validate RAG responses
- Check error handling

## Troubleshooting

Common issues and solutions:

1. Neo4j Connection:
   - Verify Neo4j is running
   - Check connection credentials
   - Ensure database name is correct

2. Document Processing:
   - Check file format compatibility
   - Verify file size limits
   - Look for encoding issues

3. RAG Queries:
   - Verify Gemini API key
   - Check context length
   - Monitor response quality

4. PyTorch/Streamlit Issues:
   - If you encounter event loop errors:
     ```
     RuntimeError: no running event loop
     ```
     Try running with python -m: `python -m streamlit run app/main.py`
   
   - If you see torch custom class errors:
     ```
     RuntimeError: Tried to instantiate class '__path__._path'
     ```
     Restart your Streamlit server and ensure no other instances are running
   
   - Component initialization errors:
     - Make sure Neo4j is running and accessible
     - Verify sentence-transformers model is properly downloaded
     - Check network connectivity for model downloads
   
   - If the app becomes unresponsive:
     - The application uses session state to manage components
     - Sometimes race conditions can occur during initialization
     - Close the browser tab and restart the application
     - If problems persist, clear your browser cache

5. Resource Management:
   - The application manages resources through Streamlit's session state
   - Resources are automatically cleaned up on page refresh
   - To ensure proper cleanup:
     1. Let the page fully load when starting
     2. Allow page refreshes to complete
     3. Use Ctrl+C to stop the server when done
   
   - If resources aren't properly cleaned up:
     1. Stop the Streamlit server (Ctrl+C)
     2. Check Neo4j connections (e.g., via Neo4j Browser)
     3. Manually close any lingering connections
     4. Restart the application

6. Logging and Debugging:
   - The application uses Python's logging module with detailed formatting
   - Log levels:
     * INFO: General application flow
     * DEBUG: Detailed debugging information
     * WARNING: Potential issues
     * ERROR: Error conditions with full stack traces
   
   - To access logs:
     1. View console output when running the application
     2. Logs include timestamps and component names
     3. Full stack traces for errors help pinpoint issues
   
   - Common log messages to watch for:
     * Component initialization status
     * File processing progress
     * Query execution flow
     * Resource cleanup events

7. Session State Management:
   - Components are initialized once and stored in session state
   - Chat history and document list persist across page refreshes
   - If you encounter state-related issues:
     1. Clear your browser cache
     2. Restart the Streamlit server
     3. Verify Neo4j database state

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
