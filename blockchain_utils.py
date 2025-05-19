from web3 import Web3
import pandas as pd
import os
from token_data import ERC20_ABI, get_token_info
import datetime
import time

def connect_to_node(node_url=None):
    """
    Connects to the Coinbase Cloud node and returns the web3 instance
    """
    if node_url is None:
        node_url = os.getenv("COINBASE_CLOUD_NODE")
        
    if not node_url:
        raise ValueError("No Coinbase Cloud node URL provided")
        
    w3 = Web3(Web3.HTTPProvider(node_url))
    
    # Check connection
    if not w3.is_connected():
        raise ConnectionError("Failed to connect to the Ethereum node")
        
    return w3

def validate_ethereum_address(address):
    """
    Validates if the given address is a valid Ethereum address
    """
    try:
        return Web3.is_address(address)
    except:
        return False

def get_eth_balance(w3, address):
    """
    Get ETH balance for an address
    """
    try:
        if not validate_ethereum_address(address):
            raise ValueError("Invalid Ethereum address")
            
        balance = w3.eth.get_balance(address)
        return w3.from_wei(balance, 'ether')
    except Exception as e:
        raise Exception(f"Error getting ETH balance: {str(e)}")

def get_token_balance(w3, address, token_symbol):
    """
    Get token balance for a specific ERC-20 token and address
    """
    try:
        if not validate_ethereum_address(address):
            raise ValueError("Invalid Ethereum address")
            
        token_info = get_token_info(token_symbol)
        
        if not token_info:
            raise ValueError(f"Token {token_symbol} not supported")
            
        token_contract = w3.eth.contract(address=token_info["address"], abi=ERC20_ABI)
        token_balance = token_contract.functions.balanceOf(address).call()
        
        # Convert based on token decimals
        return token_balance / (10 ** token_info["decimals"])
    except Exception as e:
        raise Exception(f"Error getting {token_symbol} balance: {str(e)}")

def get_token_transfers(w3, address, token_symbol, from_block, to_block=None):
    """
    Get token transfers for an address in a specific block range
    Returns inflow, outflow and a dataframe of transfers
    """
    try:
        if not validate_ethereum_address(address):
            raise ValueError("Invalid Ethereum address")
            
        token_info = get_token_info(token_symbol)
        
        if not token_info:
            raise ValueError(f"Token {token_symbol} not supported")
            
        if to_block is None:
            to_block = w3.eth.block_number
            
        token_contract = w3.eth.contract(address=token_info["address"], abi=ERC20_ABI)
        
        # Define transfer filters
        inflow_filter = token_contract.events.Transfer.create_filter(
            fromBlock=from_block, 
            toBlock=to_block, 
            argument_filters={'to': address}
        )
        
        outflow_filter = token_contract.events.Transfer.create_filter(
            fromBlock=from_block, 
            toBlock=to_block, 
            argument_filters={'from': address}
        )
        
        # Get transfer events
        inflow_events = inflow_filter.get_all_entries()
        outflow_events = outflow_filter.get_all_entries()
        
        # Process inflow events
        inflow_data = []
        total_inflow = 0
        for event in inflow_events:
            value = event['args']['value'] / (10 ** token_info["decimals"])
            total_inflow += value
            
            block = w3.eth.get_block(event['blockNumber'])
            timestamp = datetime.datetime.fromtimestamp(block['timestamp'])
            
            inflow_data.append({
                'timestamp': timestamp,
                'block': event['blockNumber'],
                'from': event['args']['from'],
                'to': event['args']['to'],
                'value': value,
                'type': 'inflow'
            })
            
        # Process outflow events
        outflow_data = []
        total_outflow = 0
        for event in outflow_events:
            value = event['args']['value'] / (10 ** token_info["decimals"])
            total_outflow += value
            
            block = w3.eth.get_block(event['blockNumber'])
            timestamp = datetime.datetime.fromtimestamp(block['timestamp'])
            
            outflow_data.append({
                'timestamp': timestamp,
                'block': event['blockNumber'],
                'from': event['args']['from'],
                'to': event['args']['to'],
                'value': value,
                'type': 'outflow'
            })
            
        # Combine and create dataframe
        all_data = inflow_data + outflow_data
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            df = df.sort_values(by='timestamp', ascending=False)
            
        return {
            'inflow': total_inflow,
            'outflow': total_outflow,
            'net_flow': total_inflow - total_outflow,
            'transfers_df': df
        }
    except Exception as e:
        raise Exception(f"Error getting {token_symbol} transfers: {str(e)}")

def get_block_info(w3, block_number='latest'):
    """
    Get information about a specific block
    """
    try:
        if block_number == 'latest':
            block_number = w3.eth.block_number
            
        block = w3.eth.get_block(block_number)
        timestamp = datetime.datetime.fromtimestamp(block['timestamp'])
        
        return {
            'number': block['number'],
            'timestamp': timestamp,
            'transactions': len(block['transactions']),
            'gas_used': block['gasUsed'],
            'gas_limit': block['gasLimit']
        }
    except Exception as e:
        raise Exception(f"Error getting block info: {str(e)}")
