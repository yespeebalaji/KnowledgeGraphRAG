import google.generativeai as genai
import logging
from typing import List, Dict, Any, Optional
import uuid
import os
import sys
import importlib.util

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
KnowledgeGraph = import_local_module("knowledge_graph").KnowledgeGraph
EmbeddingManager = import_local_module("embeddings").EmbeddingManager

# Extract config variables
GOOGLE_API_KEY = config.GOOGLE_API_KEY
MAX_CONTEXT_LENGTH = config.MAX_CONTEXT_LENGTH
TOP_K_RESULTS = config.TOP_K_RESULTS
VECTOR_SIMILARITY_THRESHOLD = config.VECTOR_SIMILARITY_THRESHOLD

class RAGEngine:
    """Manages the RAG (Retrieval-Augmented Generation) system."""

    def __init__(self):
        """Initialize RAG components."""
        try:
            api_key = GOOGLE_API_KEY
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            logging.info("Initializing Gemini with API key...")
            genai.configure(api_key=api_key)

            # Configure Gemini
            try:
                # Initialize with error handling
                logging.info("Initializing Gemini API...")
                genai.configure(api_key=api_key)
                
                # Get available models
                models = genai.list_models()
                model_names = [model.name for model in models]
                logging.debug(f"Available models: {model_names}")
                
                # Use the experimental model that's known to work
                model_name = 'models/gemini-2.0-pro-exp-02-05'
                if model_name not in model_names:
                    raise ValueError(f"Required model {model_name} not found in available models")
                
                logging.info(f"Initializing {model_name}...")
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    safety_settings=None  # Disable safety filters for testing
                )
                
                # Verify model works
                test_response = self.model.generate_content("Test.")
                if not hasattr(test_response, 'text') or not test_response.text.strip():
                    raise ValueError("Model returned empty response")
                
                logging.info("Gemini model initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Gemini: {str(e)}")
                raise

            # Initialize other components
            self.knowledge_graph = KnowledgeGraph()
            self.embedding_manager = EmbeddingManager()

            logging.info("RAG Engine initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RAG Engine: {str(e)}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and generate a response."""
        try:
            # Log incoming query
            logging.info(f"Processing query: '{query}'")
            
            # Generate query embedding
            logging.info("Generating query embedding...")
            query_embedding = self.embedding_manager.generate_embedding(query)
            
            # Perform hybrid retrieval
            logging.info("Retrieving context...")
            context = self._retrieve_context(query, query_embedding)
            logging.info(f"Retrieved context: {len(context['chunks'])} chunks, "
                        f"{context['vector_count']} from vector search, "
                        f"{context['graph_count']} from graph search")
            
            # Generate response
            logging.info("Generating LLM response...")
            response = self._generate_response(query, context)
            
            # Log the result
            logging.info(f"Generated response: '{response}'")
            
            return {
                'response': response,
                'context': context,
                'query_embedding': query_embedding
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise

    def _retrieve_context(self, query: str, query_embedding: List[float]) -> Dict[str, Any]:
        """Retrieve relevant context using hybrid approach."""
        try:
            # Get similar chunks using vector similarity
            vector_results = self.knowledge_graph.semantic_search(
                query_embedding,
                similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
                limit=TOP_K_RESULTS
            )

            # Extract entities from query for graph-based retrieval
            entities = self._extract_entities(query)
            graph_results = []
            if entities:
                graph_results = self.knowledge_graph.get_relevant_chunks(entities)

            # Combine and deduplicate results
            all_chunks = self._combine_results(vector_results, graph_results)

            return {
                'chunks': all_chunks,
                'vector_count': len(vector_results),
                'graph_count': len(graph_results)
            }
        except Exception as e:
            logging.error(f"Error retrieving context: {str(e)}")
            raise

    def _generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a response using Gemini Pro."""
        try:
            # Log context information
            logging.info(f"Preparing context from {len(context.get('chunks', []))} chunks...")
            
            # Prepare context string from chunks
            context_str = "\n\n".join([chunk.get('text', '') for chunk in context.get('chunks', [])])
            prompt = f"""Based on the following context, please answer the question.
            If you cannot answer the question based on the provided context, say so.

            Context:
            {context_str}

            Question: {query}

            Answer:"""
            
            # Log prompt length
            logging.info(f"Generated prompt with length: {len(prompt)} characters")
            
            # Generate response with safety checks
            try:
                # Log LLM request
                logging.info("Sending request to Gemini model...")
                response = self.model.generate_content(prompt)
                
                if not response or not hasattr(response, 'text'):
                    raise Exception("Empty or invalid response from Gemini")
                
                result = response.text.strip()
                logging.info(f"Received LLM response: '{result}'")
                return result
            except Exception as e:
                logging.error(f"Error generating LLM response: {str(e)}")
                error_msg = "I apologize, but I encountered an error while generating a response. Please try again."
                logging.info(f"Returning error message to user: '{error_msg}'")
                return error_msg
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            raise

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        Note: This is a simplified version. In production, use a proper NER model.
        """
        # TODO: Implement proper entity extraction
        # Simplified entity extraction (splitting on spaces and checking for capitalization)
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 1]  # Basic check
        return entities

    def _combine_results(self, vector_results: List[Dict],
                        graph_results: List[Dict]) -> List[Dict]:
        """Combine and deduplicate results from vector and graph retrieval."""
        # Use a set to track unique chunk IDs
        seen_chunks = set()
        combined_results = []

        # Process results (handling potential missing keys)
        for result in vector_results + graph_results:
            chunk_id = result.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                combined_results.append(result)

        # Sort by similarity if available, otherwise, keep the original order
        combined_results.sort(
            key=lambda x: x.get('similarity', -1),  # Use -1 as default to maintain order
            reverse=True
        )

        return combined_results[:TOP_K_RESULTS]

    def add_document(self, content: str, doc_type: str,
                    metadata: Optional[Dict] = None) -> str:
        """
        Process and add a document to the knowledge graph.

        Args:
            content: Document content
            doc_type: Type of document ('pdf', 'url', 'text')
            metadata: Optional document metadata

        Returns:
            Document ID
        """
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())

            # Add document to graph
            self.knowledge_graph.add_document(doc_id, metadata or {})

            # Process document content into chunks (improved chunking)
            chunks = self._chunk_text(content)

            # Process each chunk
            for i, chunk_text in enumerate(chunks):
                # Generate chunk ID
                chunk_id = f"{doc_id}_{i}"

                # Generate embedding
                embedding = self.embedding_manager.generate_embedding(chunk_text)

                # Add chunk to graph
                self.knowledge_graph.add_chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    metadata={'index': i},
                    embedding=embedding
                )

                # Extract and add entities (handling potential errors)
                try:
                    entities = self._extract_entities(chunk_text)
                    for entity in entities:
                        self.knowledge_graph.add_entity(
                            entity=entity,
                            entity_type='UNKNOWN',  # Consider adding proper entity typing
                            chunk_ids=[chunk_id]
                        )
                except Exception as e:
                    logging.error(f"Error extracting entities from chunk {chunk_id}: {e}")

            return doc_id
        except Exception as e:
            logging.error(f"Error adding document: {str(e)}")
            raise
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Splits the text into chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
            if start >= len(text): # prevents infinite loop if overlap >= chunk_size
              break
        return chunks


    def close(self):
        """Clean up resources."""
        self.knowledge_graph.close()
