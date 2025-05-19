"""
AI-powered insights generator for blockchain trends
"""
import os
import logging
import json
from openai import OpenAI

# Initialize OpenAI client with API key from environment
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(api_key=api_key)

def generate_blockchain_insights(blockchain_data, chain_id=None):
    """
    Generate AI-powered insights from blockchain data
    
    Parameters:
    - blockchain_data: Dictionary containing various blockchain metrics
    - chain_id: Optional chain ID to provide context
    
    Returns:
    - Dictionary with insights, recommendations, and trend analysis
    """
    try:
        # Format the data for the AI
        formatted_data = format_data_for_ai(blockchain_data, chain_id)
        
        client = get_openai_client()
        
        # Create completion with GPT-4o model
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a blockchain data analyst with expertise in on-chain metrics and forward-looking indicators.
                    Analyze the provided blockchain data and generate insights about market trends, network health, and potential future developments.
                    Be specific, data-driven, and concise. Provide clear narrative explanations and potential implications.
                    Focus on identifying patterns that might precede market movements.
                    Organize your response in JSON format with clear sections for summary, market_trends, network_health, key_indicators, and recommendations."""
                },
                {
                    "role": "user",
                    "content": f"Please analyze this blockchain data and provide insights in JSON format about what it suggests for future trends. Include 'json' in your response:\n\n{formatted_data}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=1000
        )
        
        # Parse and return the insights
        content = response.choices[0].message.content
        if content is not None:
            try:
                insights = json.loads(content)
                return insights
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse AI response as JSON",
                    "message": "The AI response was not in valid JSON format. Please try again."
                }
        else:
            return {
                "error": "No content received from AI model",
                "message": "Please try again with different parameters."
            }
    
    except Exception as e:
        logging.error(f"Error generating blockchain insights: {str(e)}")
        return {
            "error": f"Failed to generate insights: {str(e)}",
            "message": "To generate AI-powered insights, please ensure you have provided a valid OpenAI API key."
        }

def format_data_for_ai(blockchain_data, chain_id=None):
    """Format blockchain data into a structured format for the AI model"""
    
    # Network health metrics
    network_metrics = blockchain_data.get('network_metrics', {}) or {}
    
    # DeFi activity metrics
    defi_metrics = blockchain_data.get('defi_metrics', {}) or {}
    
    # Address activity metrics
    address_metrics = blockchain_data.get('address_metrics', {}) or {}
    
    # Create a formatted data structure
    data = {
        "chain_info": {
            "chain_id": chain_id,
            "timestamp": blockchain_data.get('timestamp')
        },
        "network_metrics": {
            "transaction_growth_rate": network_metrics.get('tx_growth_rate'),
            "avg_gas_price": network_metrics.get('avg_gas_price'),
            "network_congestion": network_metrics.get('avg_gas_utilization'),
            "avg_block_time": network_metrics.get('avg_block_time'),
            "avg_transactions_per_block": network_metrics.get('avg_tx_count')
        },
        "defi_metrics": {
            "total_activity": defi_metrics.get('total_activity'),
            "protocol_activity": defi_metrics.get('protocol_activity'),
            "market_concentration": calculate_market_concentration(defi_metrics.get('market_shares', {}) or {})
        },
        "address_metrics": {
            "active_addresses": address_metrics.get('current_active_addresses'),
            "address_growth_rate": address_metrics.get('active_address_growth'),
            "daily_active_trends": address_metrics.get('daily_active_addresses', {}) or {}
        }
    }
    
    return json.dumps(data, indent=2, default=str)

def calculate_market_concentration(market_shares):
    """Calculate Herfindahl-Hirschman Index (HHI) to measure market concentration"""
    if not market_shares:
        return None
        
    # HHI is the sum of squared market shares
    hhi = sum((share/100)**2 for share in market_shares.values())
    
    # Classify concentration
    if hhi < 0.15:
        return {"score": hhi, "classification": "Low concentration (competitive market)"}
    elif hhi < 0.25:
        return {"score": hhi, "classification": "Moderate concentration"}
    else:
        return {"score": hhi, "classification": "High concentration (potential centralization)"}

def get_trend_direction(values_list):
    """Determine trend direction from a list of values"""
    if not values_list or len(values_list) < 2:
        return "Stable"
        
    first_half = values_list[:len(values_list)//2]
    second_half = values_list[len(values_list)//2:]
    
    first_avg = sum(first_half) / len(first_half) if first_half else 0
    second_avg = sum(second_half) / len(second_half) if second_half else 0
    
    change = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
    
    if change > 10:
        return "Strongly Increasing"
    elif change > 3:
        return "Moderately Increasing"
    elif change < -10:
        return "Strongly Decreasing"
    elif change < -3:
        return "Moderately Decreasing"
    else:
        return "Stable"