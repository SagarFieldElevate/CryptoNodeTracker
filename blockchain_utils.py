from web3 import Web3
import pandas as pd
import os
from token_data import ERC20_ABI, get_token_info
import datetime
import time
import numpy as np

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
        elif isinstance(block_number, str) and block_number.isdigit():
            block_number = int(block_number)
            
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
        
def get_network_metrics(w3, blocks_back=100):
    """
    Get network-wide metrics that can be used as forward-looking indicators
    """
    try:
        latest_block = w3.eth.block_number
        start_block = max(0, latest_block - blocks_back)
        
        # Gather data across blocks
        block_data = []
        gas_prices = []
        transaction_counts = []
        timestamps = []
        
        for block_num in range(start_block, latest_block + 1):
            try:
                block = w3.eth.get_block(block_num)
                block_time = datetime.datetime.fromtimestamp(block['timestamp'])
                
                # Get transaction data
                tx_count = len(block['transactions'])
                transaction_counts.append(tx_count)
                
                # Get gas data
                gas_used = block['gasUsed']
                gas_limit = block['gasLimit']
                
                # Get gas price for each transaction in the block
                if tx_count > 0:
                    for tx_hash in block['transactions'][:min(5, tx_count)]:  # Sample up to 5 txs per block
                        try:
                            tx = w3.eth.get_transaction(tx_hash)
                            gas_price = tx.get('gasPrice', 0)
                            if gas_price > 0:
                                gas_prices.append(w3.from_wei(gas_price, 'gwei'))
                        except:
                            continue
                
                # Store block data
                block_data.append({
                    'block_number': block_num,
                    'timestamp': block_time,
                    'transaction_count': tx_count,
                    'gas_used': gas_used,
                    'gas_limit': gas_limit,
                    'gas_utilization': gas_used / gas_limit if gas_limit > 0 else 0
                })
                
                timestamps.append(block_time)
                
            except Exception as e:
                # Skip problematic blocks
                continue
        
        # Create dataframe
        df = pd.DataFrame(block_data)
        if df.empty:
            raise ValueError("No valid block data collected")
        
        # Compute metrics
        avg_gas_price = np.mean(gas_prices) if gas_prices else 0
        avg_tx_count = np.mean(transaction_counts) if transaction_counts else 0
        avg_gas_utilization = df['gas_utilization'].mean() if not df.empty else 0
        
        # Calculate transaction growth rate (as percentage)
        if len(transaction_counts) > 1:
            first_half = transaction_counts[:len(transaction_counts)//2]
            second_half = transaction_counts[len(transaction_counts)//2:]
            tx_growth = ((np.mean(second_half) - np.mean(first_half)) / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0
        else:
            tx_growth = 0
            
        # Calculate average block time
        block_times = []
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if 0 < time_diff < 60:  # Filter out outliers
                block_times.append(time_diff)
        
        avg_block_time = np.mean(block_times) if block_times else 0
        
        return {
            'avg_gas_price': avg_gas_price,
            'avg_tx_count': avg_tx_count, 
            'avg_gas_utilization': avg_gas_utilization * 100,  # As percentage
            'tx_growth_rate': tx_growth,
            'avg_block_time': avg_block_time,
            'block_data': df
        }
    except Exception as e:
        raise Exception(f"Error getting network metrics: {str(e)}")
        
def get_defi_indicators(w3, blocks_back=1000):
    """
    Get DeFi-specific indicators from common DeFi protocols
    """
    try:
        # This function would connect to various DeFi protocols to get their metrics
        # For demonstration, we'll use transaction data as a proxy for DeFi activity
        
        latest_block = w3.eth.block_number
        start_block = max(0, latest_block - blocks_back)
        
        # Sample some popular DeFi contract addresses
        defi_addresses = {
            'Uniswap V3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'Aave V2': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
            'Compound': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B',
            'SushiSwap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'
        }
        
        # Collect data
        defi_data = {}
        for protocol, address in defi_addresses.items():
            try:
                # Count transactions to the protocol in the last X blocks
                tx_count = 0
                for block_num in range(start_block, latest_block + 1, max(1, (latest_block - start_block) // 20)):  # Sample blocks
                    try:
                        block = w3.eth.get_block(block_num, full_transactions=True)
                        # Count transactions to this DeFi protocol
                        for tx in block['transactions']:
                            if tx['to'] and tx['to'].lower() == address.lower():
                                tx_count += 1
                    except:
                        continue
                        
                defi_data[protocol] = tx_count
            except:
                defi_data[protocol] = 0
                
        # Calculate total DeFi activity
        total_activity = sum(defi_data.values())
        
        # Calculate market share for each protocol
        market_shares = {}
        for protocol, count in defi_data.items():
            market_shares[protocol] = (count / total_activity * 100) if total_activity > 0 else 0
            
        return {
            'total_defi_activity': total_activity,
            'protocol_activity': defi_data,
            'market_shares': market_shares
        }
    except Exception as e:
        raise Exception(f"Error getting DeFi indicators: {str(e)}")
        
def get_address_activity_trends(w3, days=7):
    """
    Analyze active addresses trends over time
    """
    try:
        latest_block = w3.eth.block_number
        
        # Estimate blocks per day (avg 13.5 seconds per block)
        blocks_per_day = int(24 * 60 * 60 / 13.5)
        total_blocks = blocks_per_day * days
        
        # Limit to a reasonable number to avoid timeouts
        max_blocks = min(5000, total_blocks)
        
        # Calculate step size to evenly distribute sampling
        step = max(1, int(total_blocks / max_blocks))
        
        # Start block
        start_block = max(0, latest_block - total_blocks)
        
        # Track unique addresses
        active_addresses = set()
        daily_active = {}
        current_day = None
        
        # Sample blocks
        for block_num in range(start_block, latest_block + 1, step):
            try:
                block = w3.eth.get_block(block_num)
                timestamp = datetime.datetime.fromtimestamp(block['timestamp'])
                day = timestamp.date()
                
                if current_day is None:
                    current_day = day
                    
                # If we've moved to a new day, record the previous day's data
                if day != current_day:
                    daily_active[current_day] = len(active_addresses)
                    active_addresses = set()  # Reset for new day
                    current_day = day
                
                # For each transaction in the block
                for tx_hash in block['transactions'][:10]:  # Limit to 10 txs per block for efficiency
                    try:
                        tx = w3.eth.get_transaction(tx_hash)
                        if tx['from']:
                            active_addresses.add(tx['from'])
                        if tx['to']:
                            active_addresses.add(tx['to'])
                    except:
                        continue
            except Exception as e:
                # Skip problematic blocks
                continue
                
        # Add the last day
        if current_day and current_day not in daily_active:
            daily_active[current_day] = len(active_addresses)
            
        # Convert to dataframe
        dates = sorted(daily_active.keys())
        address_counts = [daily_active[date] for date in dates]
        
        # Calculate active address growth
        if len(address_counts) > 1:
            growth_rate = ((address_counts[-1] - address_counts[0]) / address_counts[0] * 100) if address_counts[0] > 0 else 0
        else:
            growth_rate = 0
            
        return {
            'daily_active_addresses': dict(zip([d.strftime('%Y-%m-%d') for d in dates], address_counts)),
            'active_address_growth': growth_rate,
            'current_active_addresses': address_counts[-1] if address_counts else 0
        }
    except Exception as e:
        raise Exception(f"Error getting address activity trends: {str(e)}")
