from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging
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
NEO4J_URI = config.NEO4J_URI
NEO4J_USERNAME = config.NEO4J_USERNAME
NEO4J_PASSWORD = config.NEO4J_PASSWORD
NEO4J_DATABASE = config.NEO4J_DATABASE

class KnowledgeGraph:
    """Manages Neo4j knowledge graph operations."""
    
    def __init__(self):
        """Initialize Neo4j connection with comprehensive error handling."""
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Test if Neo4j is running first
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    try:
                        s.connect(('localhost', 7687))
                    except (socket.timeout, ConnectionRefusedError):
                        raise Exception(
                            "\nNeo4j is not running! Please ensure that:\n"
                            "1. Neo4j is installed (https://neo4j.com/download/)\n"
                            "2. The Neo4j service is running\n"
                            "3. The bolt protocol is enabled on port 7687\n"
                            "4. The credentials in .env match your Neo4j settings\n\n"
                            "To start Neo4j:\n"
                            "- Windows: Start Neo4j Desktop or run 'neo4j.bat start'\n"
                            "- Linux/Mac: Run 'neo4j start' or 'brew services start neo4j'\n"
                        )

                # Attempt to connect to Neo4j
                self.driver = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                    connection_timeout=5,
                    max_connection_lifetime=60 * 60,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60
                )

                # Verify database connection and capabilities
                with self.driver.session(database=NEO4J_DATABASE) as session:
                    # Basic connectivity test
                    session.run("RETURN datetime()")
                    
                    # Test vector operations
                    try:
                        session.run("""
                        WITH [1,2,3] as v1, [4,5,6] as v2
                        RETURN reduce(dot = 0.0, i in range(0, size(v1)-1) | dot + v1[i] * v2[i]) as test
                        """)
                    except Exception as e:
                        raise Exception(
                            "\nNeo4j vector operations test failed. Please ensure that:\n"
                            "1. You're using Neo4j 4.4+ for vector operation support\n"
                            "2. The database has enough memory allocated\n"
                            "3. The database user has write permissions\n"
                            f"\nError details: {str(e)}"
                        )

                logging.info("Successfully connected to Neo4j database with vector operation support")
                break

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                logging.warning(f"Connection attempt {retry_count} failed:\n{last_error}")
                if retry_count < max_retries:
                    import time
                    time.sleep(2 ** retry_count)  # Exponential backoff
                continue
        
        if retry_count >= max_retries:
            error_msg = (
                f"\nFailed to connect to Neo4j after {max_retries} attempts.\n"
                f"Last error: {last_error}\n\n"
                "Please check:\n"
                "1. Neo4j is running and accessible\n"
                "2. Connection credentials are correct\n"
                "3. The database exists and is operational\n"
                "4. Network/firewall settings allow connection\n"
            )
            logging.error(error_msg)
            raise Exception(error_msg)
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    
    def setup_database(self):
        """Initialize database schema and constraints."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Create constraints with version check
                version_query = "CALL dbms.components() YIELD versions RETURN versions[0] as version"
                version = session.run(version_query).single()["version"]
                
                if version.startswith("5"):
                    # Neo4j 5.x constraint syntax
                    constraints = [
                        """CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document)
                           REQUIRE d.id IS NOT NULL AND d.id IS UNIQUE""",
                        """CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk)
                           REQUIRE c.id IS NOT NULL AND c.id IS UNIQUE""",
                        """CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity)
                           REQUIRE e.name IS NOT NULL AND e.name IS UNIQUE"""
                    ]
                    for constraint in constraints:
                        session.run(constraint)
                    
                    # Create vector index (Neo4j 5.x)
                    session.run("""
                        CREATE INDEX chunk_embedding IF NOT EXISTS
                        FOR (c:Chunk) ON (c.embedding)
                    """)
                else:
                    # Fallback for older versions
                    constraints = [
                        "CREATE CONSTRAINT ON (d:Document) ASSERT d.id IS UNIQUE",
                        "CREATE CONSTRAINT ON (c:Chunk) ASSERT c.id IS UNIQUE",
                        "CREATE CONSTRAINT ON (e:Entity) ASSERT e.name IS UNIQUE"
                    ]
                    for constraint in constraints:
                        try:
                            session.run(f"{constraint} IF NOT EXISTS")
                        except:
                            try:
                                session.run(constraint)
                            except:
                                logging.warning(f"Could not create constraint: {constraint}")
                logging.info("Database schema initialized successfully")
        except Exception as e:
            logging.error(f"Failed to setup database schema: {str(e)}")
            raise
    
    def add_document(self, doc_id: str, metadata: Dict) -> None:
        """Add a document node to the graph."""
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d += $metadata
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(query, doc_id=doc_id, metadata=metadata)
    
    def add_chunk(self, chunk_id: str, doc_id: str, text: str, 
                  metadata: Dict, embedding: List[float]) -> None:
        """Add a text chunk node and connect it to its document."""
        query = """
        MATCH (d:Document {id: $doc_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.embedding = $embedding,
            c += $metadata
        MERGE (c)-[:BELONGS_TO]->(d)
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(query, chunk_id=chunk_id, doc_id=doc_id, 
                       text=text, embedding=embedding, metadata=metadata)
    
    def add_entity(self, entity: str, entity_type: str, 
                   chunk_ids: List[str]) -> None:
        """Add an entity node and connect it to relevant chunks."""
        query = """
        MERGE (e:Entity {name: $entity})
        SET e.type = $entity_type
        WITH e
        UNWIND $chunk_ids as chunk_id
        MATCH (c:Chunk {id: chunk_id})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(query, entity=entity, entity_type=entity_type, 
                       chunk_ids=chunk_ids)
    
    def add_relationship(self, entity1: str, entity2: str, 
                        relationship_type: str, metadata: Optional[Dict] = None) -> None:
        """Add a relationship between two entities."""
        query = """
        MATCH (e1:Entity {name: $entity1})
        MATCH (e2:Entity {name: $entity2})
        MERGE (e1)-[r:RELATES_TO {type: $relationship_type}]->(e2)
        SET r += $metadata
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(query, entity1=entity1, entity2=entity2,
                       relationship_type=relationship_type, 
                       metadata=metadata or {})
    
    def get_relevant_chunks(self, entity_names: List[str], 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve chunks relevant to given entities."""
        query = """
        UNWIND $entity_names as entity_name
        MATCH (e:Entity {name: entity_name})-[:MENTIONED_IN]->(c:Chunk)
        RETURN DISTINCT c.text as text, c.id as chunk_id, c.embedding as embedding
        LIMIT $limit
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, entity_names=entity_names, limit=limit)
            return [dict(record) for record in result]
    
    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get context information for an entity."""
        query = """
        MATCH (e:Entity {name: $entity_name})
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(related:Entity)
        RETURN e.name as name,
               e.type as type,
               collect(DISTINCT {
                   name: related.name,
                   type: related.type,
                   relationship: r.type
               }) as relationships
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, entity_name=entity_name)
            return dict(result.single())
    
    def semantic_search(self, query_embedding: List[float], 
                       similarity_threshold: float = 0.75,
                       limit: int = 5) -> List[Dict[str, Any]]:
        """Find chunks similar to the query embedding using batched processing."""
        # Pre-compute query magnitude
        query_magnitude = (sum(x * x for x in query_embedding)) ** 0.5

        # Neo4j 5.x optimized vector similarity query
        query = """
        MATCH (c:Chunk)
        WITH c, $query_embedding AS qe, $query_magnitude AS qm,
             sqrt(reduce(l2 = 0.0, i IN range(0, size(c.embedding)-1) | 
                l2 + c.embedding[i] * c.embedding[i])) AS cm
        WITH c,
             reduce(dot = 0.0, i IN range(0, size(qe)-1) | 
                dot + c.embedding[i] * qe[i]) / (qm * cm) AS similarity
        WHERE similarity > $threshold
        RETURN c.text AS text,
               c.id AS chunk_id,
               similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, 
                               query_embedding=query_embedding,
                               query_magnitude=query_magnitude,
                               threshold=similarity_threshold, 
                               limit=limit)
            return [dict(record) for record in result]
    
    def cleanup(self):
        """Remove all nodes and relationships from the graph."""
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(query)
