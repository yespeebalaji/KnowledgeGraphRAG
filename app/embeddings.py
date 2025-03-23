from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
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

# Import local configuration
config = import_local_module("config")
EMBEDDING_MODEL = config.EMBEDDING_MODEL
import logging

class EmbeddingManager:
    """Manages text embeddings using SentenceTransformers."""
    
    def __init__(self):
        """Initialize the embedding model."""
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logging.info(f"Successfully loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            # Generate embedding and convert to list of floats
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error generating embeddings batch: {str(e)}")
            raise
    
    def compute_similarity(self, embedding1: List[float], 
                         embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Convert lists to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except Exception as e:
            logging.error(f"Error computing similarity: {str(e)}")
            raise
    
    def batch_compute_similarity(self, query_embedding: List[float], 
                               candidate_embeddings: List[List[float]]) -> List[float]:
        """Compute similarities between a query embedding and multiple candidates."""
        try:
            # Convert to numpy arrays
            query_vec = np.array(query_embedding)
            candidate_vecs = np.array(candidate_embeddings)
            
            # Compute similarities
            similarities = np.dot(candidate_vecs, query_vec) / (
                np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_vec)
            )
            return similarities.tolist()
        except Exception as e:
            logging.error(f"Error computing batch similarities: {str(e)}")
            raise
    
    def find_most_similar(self, query_embedding: List[float],
                         candidate_embeddings: List[List[float]],
                         texts: List[str],
                         top_k: int = 5) -> List[Dict]:
        """Find the most similar texts based on embeddings."""
        try:
            # Compute similarities
            similarities = self.batch_compute_similarity(query_embedding, candidate_embeddings)
            
            # Create list of (similarity, text) tuples
            results = list(zip(similarities, texts))
            
            # Sort by similarity in descending order
            results.sort(key=lambda x: x[0], reverse=True)
            
            # Return top k results
            return [
                {'text': text, 'similarity': float(sim)}
                for sim, text in results[:top_k]
            ]
        except Exception as e:
            logging.error(f"Error finding most similar texts: {str(e)}")
            raise
    
    def average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Compute the average of multiple embeddings."""
        try:
            # Convert to numpy array and compute mean
            avg_embedding = np.mean(np.array(embeddings), axis=0)
            return avg_embedding.tolist()
        except Exception as e:
            logging.error(f"Error computing average embedding: {str(e)}")
            raise
