"""
Vector database integration for storing blockchain analytics data
"""
import os
import json
import uuid
import logging
from datetime import datetime
import numpy as np
from pinecone import Pinecone
from sklearn.preprocessing import normalize

# Initialize Pinecone client with API key
def get_pinecone_client():
    """Get authenticated Pinecone client"""
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    return Pinecone(api_key=api_key)

def create_index_if_not_exists(pc, index_name="blockchain-analytics", dimension=1536):
    """Create a Pinecone index if it doesn't already exist"""
    try:
        # List existing indexes
        indexes = pc.list_indexes()
        
        # Check if our index already exists
        if index_name not in indexes:
            logging.info(f"Creating new Pinecone index: {index_name}")
            
            # Create a new index
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            logging.info(f"Successfully created index {index_name}")
        else:
            logging.info(f"Using existing index {index_name}")
            
        # Connect to the index
        return pc.Index(index_name)
    
    except Exception as e:
        logging.error(f"Error creating/accessing Pinecone index: {str(e)}")
        raise

def generate_simple_embedding(data_dict, dimension=1536):
    """
    Generate a simple embedding for a data dictionary
    This is a placeholder for a more sophisticated embedding method
    """
    # Flatten the dictionary to a string
    data_str = json.dumps(data_dict)
    
    # Generate a deterministic embedding based on the hash of the string
    # This is just for demonstration - in a real app, you'd use proper embeddings
    hash_val = hash(data_str) % 10000000
    
    # Convert hash to a pseudo-random vector
    np.random.seed(hash_val)
    embedding = np.random.random(dimension).astype(np.float32)
    
    # Normalize to unit length for cosine similarity
    return normalize(embedding.reshape(1, -1))[0].tolist()

def store_network_metrics(network_data, chain_id=None):
    """
    Store network metrics in Pinecone vector database
    
    Parameters:
    - network_data: Dictionary containing network metrics
    - chain_id: Optional chain ID to provide context
    
    Returns:
    - ID of the stored record
    """
    try:
        pc = get_pinecone_client()
        index = create_index_if_not_exists(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "network_metrics",
            "chain_id": chain_id,
            "metrics": network_data
        }
        
        # Generate embedding
        embedding = generate_simple_embedding(network_data)
        
        # Upsert record to Pinecone
        index.upsert(
            vectors=[
                {
                    "id": record_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        logging.info(f"Stored network metrics in Pinecone with ID: {record_id}")
        return record_id
    
    except Exception as e:
        logging.error(f"Error storing network metrics in Pinecone: {str(e)}")
        return None

def store_defi_metrics(defi_data, chain_id=None):
    """
    Store DeFi metrics in Pinecone vector database
    
    Parameters:
    - defi_data: Dictionary containing DeFi metrics
    - chain_id: Optional chain ID to provide context
    
    Returns:
    - ID of the stored record
    """
    try:
        pc = get_pinecone_client()
        index = create_index_if_not_exists(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "defi_metrics",
            "chain_id": chain_id,
            "metrics": defi_data
        }
        
        # Generate embedding
        embedding = generate_simple_embedding(defi_data)
        
        # Upsert record to Pinecone
        index.upsert(
            vectors=[
                {
                    "id": record_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        logging.info(f"Stored DeFi metrics in Pinecone with ID: {record_id}")
        return record_id
    
    except Exception as e:
        logging.error(f"Error storing DeFi metrics in Pinecone: {str(e)}")
        return None

def store_address_metrics(address_data, chain_id=None):
    """
    Store address activity metrics in Pinecone vector database
    
    Parameters:
    - address_data: Dictionary containing address activity metrics
    - chain_id: Optional chain ID to provide context
    
    Returns:
    - ID of the stored record
    """
    try:
        pc = get_pinecone_client()
        index = create_index_if_not_exists(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "address_metrics",
            "chain_id": chain_id,
            "metrics": address_data
        }
        
        # Generate embedding
        embedding = generate_simple_embedding(address_data)
        
        # Upsert record to Pinecone
        index.upsert(
            vectors=[
                {
                    "id": record_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        logging.info(f"Stored address metrics in Pinecone with ID: {record_id}")
        return record_id
    
    except Exception as e:
        logging.error(f"Error storing address metrics in Pinecone: {str(e)}")
        return None

def store_ai_insights(insights_data, query_context=None, chain_id=None):
    """
    Store AI-generated insights in Pinecone vector database
    
    Parameters:
    - insights_data: Dictionary containing AI insights
    - query_context: Optional context about the query that generated these insights
    - chain_id: Optional chain ID to provide context
    
    Returns:
    - ID of the stored record
    """
    try:
        pc = get_pinecone_client()
        index = create_index_if_not_exists(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "ai_insights",
            "chain_id": chain_id,
            "query_context": query_context,
            "insights": insights_data
        }
        
        # Generate embedding
        embedding = generate_simple_embedding(insights_data)
        
        # Upsert record to Pinecone
        index.upsert(
            vectors=[
                {
                    "id": record_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        logging.info(f"Stored AI insights in Pinecone with ID: {record_id}")
        return record_id
    
    except Exception as e:
        logging.error(f"Error storing AI insights in Pinecone: {str(e)}")
        return None

def store_prediction_data(prediction_data, historical_context=None, chain_id=None):
    """
    Store predictive analytics results in Pinecone vector database
    
    Parameters:
    - prediction_data: Dictionary containing prediction results
    - historical_context: Optional context about the historical data used
    - chain_id: Optional chain ID to provide context
    
    Returns:
    - ID of the stored record
    """
    try:
        pc = get_pinecone_client()
        index = create_index_if_not_exists(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "predictions",
            "chain_id": chain_id,
            "historical_context": historical_context,
            "predictions": prediction_data
        }
        
        # Generate embedding
        embedding = generate_simple_embedding(prediction_data)
        
        # Upsert record to Pinecone
        index.upsert(
            vectors=[
                {
                    "id": record_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        logging.info(f"Stored prediction data in Pinecone with ID: {record_id}")
        return record_id
    
    except Exception as e:
        logging.error(f"Error storing prediction data in Pinecone: {str(e)}")
        return None

def query_similar_data(query_data, data_type=None, top_k=5):
    """
    Query for similar data points in the vector database
    
    Parameters:
    - query_data: Dictionary or embedding to query with
    - data_type: Optional filter by data type (network_metrics, defi_metrics, etc.)
    - top_k: Number of results to return
    
    Returns:
    - List of similar data points with their metadata
    """
    try:
        pc = get_pinecone_client()
        index = create_index_if_not_exists(pc)
        
        # Generate query embedding
        if isinstance(query_data, list):
            query_vector = query_data  # Already an embedding
        else:
            query_vector = generate_simple_embedding(query_data)
        
        # Build filter if data_type is specified
        filter_dict = {"data_type": data_type} if data_type else None
        
        # Query the index
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return results.matches
    
    except Exception as e:
        logging.error(f"Error querying Pinecone: {str(e)}")
        return []

def list_recent_data(data_type=None, limit=10):
    """
    List recent data points from the vector database
    Note: This is a basic implementation and might not be efficient for large datasets
    
    Parameters:
    - data_type: Optional filter by data type
    - limit: Maximum number of results to return
    
    Returns:
    - List of recent data points with their metadata
    """
    try:
        pc = get_pinecone_client()
        index = create_index_if_not_exists(pc)
        
        # For demonstration - in a real app, you would use a more efficient approach
        # This just queries with a random vector and relies on the metadata filter
        query_vector = np.random.randn(1536).tolist()
        
        # Build filter if data_type is specified
        filter_dict = {"data_type": {"$eq": data_type}} if data_type else None
        
        # Query the index
        results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Sort by timestamp (most recent first)
        sorted_results = sorted(
            results.matches,
            key=lambda x: x.metadata.get("timestamp", ""),
            reverse=True
        )
        
        return sorted_results[:limit]
    
    except Exception as e:
        logging.error(f"Error listing recent data from Pinecone: {str(e)}")
        return []