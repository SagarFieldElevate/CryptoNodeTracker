"""
Vector database integration for storing blockchain analytics data
"""
import os
import json
import uuid
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from pinecone import Pinecone
from sklearn.preprocessing import normalize

def make_json_serializable(obj):
    """
    Convert any object to a JSON serializable format.
    Handles pandas DataFrames, NumPy arrays, datetime objects, and more.
    
    Parameters:
    - obj: Any Python object
    
    Returns:
    - JSON serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        try:
            # Try to convert to dict first
            dict_data = obj.to_dict(orient='records')
            # Make all values in the dict serializable
            return [make_json_serializable(record) for record in dict_data]
        except:
            # Fallback to string representation
            return obj.to_string()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        try:
            return make_json_serializable(obj.to_dict())
        except:
            return obj.to_string()
    elif hasattr(obj, 'to_dict'):
        # For any object with to_dict method (like some pandas objects)
        try:
            return make_json_serializable(obj.to_dict())
        except:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # For general objects with attributes
        return make_json_serializable(obj.__dict__)
    elif np.isscalar(obj) and np.isnan(obj):
        # Handle NaN values
        return None
    elif pd.isna(obj):
        # Handle pandas NA values
        return None
    else:
        # Try direct conversion, if it fails convert to string
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

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
        # Try to directly get the index first, which will work if it exists
        try:
            index = pc.Index(index_name)
            logging.info(f"Using existing index {index_name}")
            return index
        except Exception as index_error:
            # If the index doesn't exist, we'll create it
            logging.info(f"Index doesn't exist yet: {str(index_error)}")
            
            # Create a new index with the required server spec
            from pinecone import ServerlessSpec
            
            try:
                # Use us-east-1 region which is available on the free plan
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws", 
                        region="us-east-1"
                    )
                )
                logging.info(f"Successfully created index {index_name}")
                
                # Now we can connect to the newly created index
                return pc.Index(index_name)
            except Exception as create_error:
                # If we get a 409 conflict error, it means the index already exists
                if "409" in str(create_error) or "already exists" in str(create_error):
                    logging.info(f"Index creation conflict - index already exists")
                    return pc.Index(index_name)
                else:
                    # If it's another error, re-raise it
                    raise create_error
    
    except Exception as e:
        logging.error(f"Error creating/accessing Pinecone index: {str(e)}")
        # Just return None instead of raising, to allow graceful handling
        return None

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
        
        # Check if we got a valid index
        if index is None:
            logging.error("Could not access Pinecone index, skipping storage")
            return None
            
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Convert all data to JSON serializable formats using our utility function
        serializable_network_data = make_json_serializable(network_data)
        
        # Log the keys we're going to store
        if isinstance(serializable_network_data, dict):
            logging.info(f"Storing network metrics with keys: {list(serializable_network_data.keys())}")
        else:
            logging.info(f"Storing network metrics (type: {type(serializable_network_data)})")
        
        # Pinecone requires metadata values to be strings, numbers, booleans or lists of strings
        # So we'll flatten and simplify our metadata structure
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "network_metrics",
            "chain_id": str(chain_id) if chain_id else "unknown"
        }
        
        # Add some key metrics as simple values
        if isinstance(serializable_network_data, dict):
            for key in ['avg_tx_count', 'tx_growth_rate', 'avg_gas_price']:
                if key in serializable_network_data:
                    metadata[key] = str(serializable_network_data.get(key, 0))
        
        # Store the full data as a JSON string
        metadata["data_json"] = json.dumps(serializable_network_data)
        
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
        
        # Check if we got a valid index
        if index is None:
            logging.error("Could not access Pinecone index, skipping storage")
            return None
            
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Create a manually JSON-serializable version of the data
        serializable_defi_data = {}
        
        # Handle each field separately to ensure JSON compatibility
        if 'total_activity' in defi_data:
            serializable_defi_data['total_activity'] = float(defi_data['total_activity'])
            
        if 'protocol_activity' in defi_data:
            serializable_defi_data['protocol_activity'] = {
                k: float(v) for k, v in defi_data['protocol_activity'].items()
            }
            
        if 'market_shares' in defi_data:
            serializable_defi_data['market_shares'] = {
                k: float(v) for k, v in defi_data['market_shares'].items()
            }
            
        if 'transaction_history' in defi_data:
            # Handle the DataFrame by converting to a list of dicts
            tx_history = defi_data['transaction_history']
            if isinstance(tx_history, pd.DataFrame):
                # Convert DataFrame to records but handle datetime objects specially
                records = []
                for _, row in tx_history.iterrows():
                    record = {}
                    for col in tx_history.columns:
                        value = row[col]
                        if isinstance(value, (pd.Timestamp, datetime)):
                            record[col] = value.isoformat()
                        else:
                            record[col] = value
                    records.append(record)
                serializable_defi_data['transaction_history'] = records
            elif isinstance(tx_history, list):
                # If it's already a list, convert datetime objects to strings
                records = []
                for item in tx_history:
                    record = {}
                    for k, v in item.items():
                        if isinstance(v, (pd.Timestamp, datetime)):
                            record[k] = v.isoformat()
                        else:
                            record[k] = v
                    records.append(record)
                serializable_defi_data['transaction_history'] = records
            else:
                serializable_defi_data['transaction_history'] = []
                
        # Log what we're storing
        logging.info(f"Storing DeFi metrics with keys: {list(serializable_defi_data.keys())}")
        
        # Pinecone requires metadata values to be strings, numbers, booleans or lists of strings
        # So we'll flatten and simplify our metadata structure
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "defi_metrics",
            "chain_id": str(chain_id) if chain_id else "unknown",
            "total_activity": str(serializable_defi_data.get('total_activity', 0)),
            # Convert protocol activity to a simpler format
            "top_protocol": max(serializable_defi_data.get('protocol_activity', {}).items(), key=lambda x: x[1])[0] if serializable_defi_data.get('protocol_activity', {}) else "none",
            # Number of transactions in history
            "tx_count": str(len(serializable_defi_data.get('transaction_history', [])))
        }
        
        # Store the full data as a JSON string instead
        metadata["data_json"] = json.dumps(serializable_defi_data)
        
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
        
        # Check if we got a valid index
        if index is None:
            logging.error("Could not access Pinecone index, skipping storage")
            return None
            
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        # Convert any pandas DataFrames to dict for JSON serialization
        serializable_address_data = {}
        for key, value in address_data.items():
            if isinstance(value, pd.DataFrame):
                # Convert any DataFrame to dict
                serializable_address_data[key] = value.to_dict(orient='records')
            else:
                serializable_address_data[key] = value
        
        # Pinecone requires metadata values to be strings, numbers, booleans or lists of strings
        # So we'll flatten and simplify our metadata structure
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "address_metrics",
            "chain_id": str(chain_id) if chain_id else "unknown"
        }
        
        # Add some key metrics as simple values
        if isinstance(serializable_address_data, dict):
            for key in ['current_active_addresses', 'active_address_growth']:
                if key in serializable_address_data:
                    metadata[key] = str(serializable_address_data.get(key, 0))
        
        # Store the full data as a JSON string
        metadata["data_json"] = json.dumps(serializable_address_data)
        
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
        
        # Check if we got a valid index
        if index is None:
            logging.error("Could not access Pinecone index, skipping storage")
            return None
            
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        # Convert any pandas DataFrames to dict for JSON serialization
        serializable_insights_data = {}
        for key, value in insights_data.items():
            if isinstance(value, pd.DataFrame):
                # Convert any DataFrame to dict
                serializable_insights_data[key] = value.to_dict(orient='records')
            else:
                serializable_insights_data[key] = value
        
        # Pinecone requires metadata values to be strings, numbers, booleans or lists of strings
        # So we'll flatten and simplify our metadata structure
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "ai_insights",
            "chain_id": str(chain_id) if chain_id else "unknown",
            "query_context": str(query_context) if query_context else "none"
        }
        
        # Add a summary if available
        if isinstance(serializable_insights_data, dict) and 'summary' in serializable_insights_data:
            metadata["summary"] = str(serializable_insights_data.get('summary', ''))[:500]  # Limit length
        
        # Store the full data as a JSON string
        metadata["data_json"] = json.dumps(serializable_insights_data)
        
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
        
        # Check if we got a valid index
        if index is None:
            logging.error("Could not access Pinecone index, skipping storage")
            return None
            
        # Generate a unique ID for this record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and chain info
        # Convert any pandas DataFrames to dict for JSON serialization
        serializable_prediction_data = {}
        for key, value in prediction_data.items():
            if isinstance(value, pd.DataFrame):
                # Convert any DataFrame to dict
                serializable_prediction_data[key] = value.to_dict(orient='records')
            else:
                serializable_prediction_data[key] = value
                
        # Pinecone requires metadata values to be strings, numbers, booleans or lists of strings
        # So we'll flatten and simplify our metadata structure
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data_type": "predictions",
            "chain_id": str(chain_id) if chain_id else "unknown",
            "historical_context": str(historical_context) if historical_context else "none"
        }
        
        # Add some key prediction metrics if available
        if isinstance(serializable_prediction_data, dict):
            # Extract confidence score if available
            if 'confidence' in serializable_prediction_data:
                conf_data = serializable_prediction_data.get('confidence', {})
                if isinstance(conf_data, dict):
                    avg_confidence = sum(conf_data.values()) / len(conf_data) if conf_data else 0
                    metadata["avg_confidence"] = str(avg_confidence)
            
            # Extract trend direction if available
            if 'trends' in serializable_prediction_data:
                trend_data = serializable_prediction_data.get('trends', {})
                if isinstance(trend_data, dict) and trend_data:
                    # Get the most common trend direction
                    trend_counts = {}
                    for trend in trend_data.values():
                        trend_counts[trend] = trend_counts.get(trend, 0) + 1
                    main_trend = max(trend_counts.items(), key=lambda x: x[1])[0]
                    metadata["main_trend"] = main_trend
        
        # Store the full data as a JSON string
        metadata["data_json"] = json.dumps(serializable_prediction_data)
        
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
        
        # Check if we got a valid index
        if index is None:
            logging.error("Could not access Pinecone index, skipping retrieval")
            return []
            
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
        if hasattr(results, 'matches'):
            return results.matches
        else:
            # Handle different response structure
            return results.get('matches', [])
    
    except Exception as e:
        logging.error(f"Error retrieving records from Pinecone: {str(e)}")
        return []