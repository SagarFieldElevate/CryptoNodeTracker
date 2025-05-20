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

def get_index(pc, index_name="blockchain-analytics"):
    """Get or create a Pinecone index"""
    try:
        # List existing indexes
        indexes = pc.list_indexes()
        
        # Check if our index already exists
        if index_name not in indexes:
            logging.info(f"Creating new Pinecone index: {index_name}")
            
            # Create a new index
            pc.create_index(
                name=index_name,
                dimension=1536,
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
    np.random.seed(hash(data_str) % 10000000)
    embedding = np.random.random(dimension).astype(np.float32)
    
    # Normalize to unit length for cosine similarity
    return normalize(embedding.reshape(1, -1))[0].tolist()

def store_network_metrics(network_data, chain_id=None):
    """Store network metrics in vector database"""
    try:
        pc = get_pinecone_client()
        index = get_index(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "network_metrics",
            "chain_id": str(chain_id) if chain_id else None,
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
    """Store DeFi metrics in vector database"""
    try:
        pc = get_pinecone_client()
        index = get_index(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "defi_metrics",
            "chain_id": str(chain_id) if chain_id else None,
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
    """Store address activity metrics in vector database"""
    try:
        pc = get_pinecone_client()
        index = get_index(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "address_metrics",
            "chain_id": str(chain_id) if chain_id else None,
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
    """Store AI-generated insights in vector database"""
    try:
        pc = get_pinecone_client()
        index = get_index(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "ai_insights",
            "chain_id": str(chain_id) if chain_id else None,
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
    """Store predictive analytics results in vector database"""
    try:
        pc = get_pinecone_client()
        index = get_index(pc)
        
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "predictions",
            "chain_id": str(chain_id) if chain_id else None,
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

def retrieve_recent_records(data_type=None, limit=10):
    """Retrieve recent records from vector database"""
    try:
        pc = get_pinecone_client()
        index = get_index(pc)
        
        # For demonstration - we'll query with a random vector
        # In a real app, you would implement a more sophisticated approach
        query_vector = list(np.random.randn(1536))
        
        # Set filter for data_type if specified
        if data_type:
            filter_params = {"data_type": data_type}
        else:
            filter_params = None
        
        # Query the index
        results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True,
            filter=filter_params
        )
        
        # Return results
        return results.matches
    
    except Exception as e:
        logging.error(f"Error retrieving records from Pinecone: {str(e)}")
        return []