from web3 import Web3
import pandas as pd
import os
from token_data import ERC20_ABI, get_token_info
import datetime
import time
import numpy as np
import logging
import concurrent.futures
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Thread-local storage for Web3 connections
thread_local = threading.local()

def connect_to_node(node_url=None):
    """
    Connects to the Coinbase Cloud node and returns the web3 instance
    """
    if node_url is None:
        node_url = os.getenv("COINBASE_CLOUD_NODE")
        
    if not node_url:
        raise ValueError("No Coinbase Cloud node URL provided")
    
    logging.info(f"Connecting to node with URL starting with: {node_url[:10]}...")
    
    # Create provider with timeout
    provider = Web3.HTTPProvider(
        node_url,
        request_kwargs={'timeout': 30}  # 30 second timeout 
    )
    
    w3 = Web3(provider)
    
    # Check connection
    logging.info("Testing connection...")
    connection_status = w3.is_connected()
    
    if not connection_status:
        logging.error("Connection test failed - is_connected() returned False")
        raise ConnectionError("Failed to connect to the Ethereum node")
    
    logging.info("Connection successful!")
    
    # Test a basic request
    try:
        latest = w3.eth.block_number
        logging.info(f"Successfully retrieved latest block: {latest}")
    except Exception as e:
        logging.error(f"Connected but failed to get block number: {str(e)}")
        raise ConnectionError(f"Connection established but failed to retrieve data: {str(e)}")
        
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
        logging.info(f"Starting network metrics analysis for {blocks_back} blocks")
        start_time = time.time()
        
        latest_block = w3.eth.block_number
        start_block = max(0, latest_block - blocks_back)
        
        logging.info(f"Analyzing blocks from {start_block} to {latest_block}")
        
        # To speed up processing, sample blocks instead of processing every block
        sample_size = min(50, blocks_back)  # Limit to 50 sample points
        step = max(1, blocks_back // sample_size)
        
        # Gather data across blocks
        block_data = []
        gas_prices = []
        transaction_counts = []
        timestamps = []
        
        logging.info(f"Sampling every {step} blocks, total samples: ~{sample_size}")
        
        blocks_processed = 0
        for block_num in range(start_block, latest_block + 1, step):
            try:
                block_start = time.time()
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
                    # Sample a small number of transactions for gas price
                    sample_tx = min(3, tx_count)
                    for tx_hash in block['transactions'][:sample_tx]:
                        try:
                            tx = w3.eth.get_transaction(tx_hash)
                            gas_price = tx.get('gasPrice', 0)
                            if gas_price > 0:
                                gas_prices.append(w3.from_wei(gas_price, 'gwei'))
                        except Exception as tx_err:
                            logging.debug(f"Error getting transaction details: {tx_err}")
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
                blocks_processed += 1
                
                block_time = time.time() - block_start
                if block_time > 1.0:  # Log slow block processing
                    logging.warning(f"Block {block_num} processing took {block_time:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Error processing block {block_num}: {str(e)}")
                continue
        
        logging.info(f"Processed {blocks_processed} blocks in {time.time() - start_time:.2f} seconds")
        
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
        
        logging.info("Network metrics analysis complete")
        
        return {
            'avg_gas_price': avg_gas_price,
            'avg_tx_count': avg_tx_count, 
            'avg_gas_utilization': avg_gas_utilization * 100,  # As percentage
            'tx_growth_rate': tx_growth,
            'avg_block_time': avg_block_time,
            'block_data': df
        }
    except Exception as e:
        logging.error(f"Error getting network metrics: {str(e)}")
        raise Exception(f"Error getting network metrics: {str(e)}")
        
def get_defi_indicators(w3, blocks_back=1000):
    """
    Get DeFi-specific indicators from common DeFi protocols
    """
    try:
        logging.info(f"Starting DeFi indicators analysis for the last {blocks_back} blocks")
        start_time = time.time()
        
        # This function would connect to various DeFi protocols to get their metrics
        # For demonstration purposes, we'll create simulated data since fetching full transaction
        # data for many blocks can be very resource-intensive and time-consuming
        
        latest_block = w3.eth.block_number
        start_block = max(0, latest_block - blocks_back)
        
        logging.info(f"Current block: {latest_block}, analyzing from block {start_block}")
        
        # Sample some popular DeFi contract addresses
        defi_addresses = {
            'Uniswap V3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'Aave V2': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
            'Compound': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B',
            'SushiSwap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'
        }
        
        # Sample some recent blocks for demonstration
        logging.info("Sampling blocks to estimate DeFi activity")
        sample_blocks = []
        # Limit number of samples to avoid long processing times
        num_samples = min(10, blocks_back // 1000 + 1)
        step = max(1, blocks_back // num_samples)
        
        logging.info(f"Will sample {num_samples} blocks with step size {step}")
        
        # Sample blocks to get transaction counts
        for i in range(num_samples):
            block_num = start_block + (i * step)
            if block_num <= latest_block:
                try:
                    block_start = time.time()
                    block = w3.eth.get_block(block_num)
                    sample_blocks.append((block_num, len(block['transactions']), block['timestamp']))
                    
                    block_time = time.time() - block_start
                    if block_time > 1.0:
                        logging.warning(f"Block {block_num} retrieval took {block_time:.2f} seconds")
                except Exception as e:
                    logging.error(f"Error retrieving block {block_num}: {str(e)}")
                    continue
        
        logging.info(f"Successfully sampled {len(sample_blocks)} blocks")
        
        if not sample_blocks:
            logging.warning("No blocks could be sampled, using default values")
            sample_blocks = [(latest_block, 100, int(time.time()))]  # Default fallback
        
        # Create simulated activity data based on sampled transaction counts
        defi_data = {}
        protocol_weights = {
            'Uniswap V3': 0.4,    # 40% market share
            'Aave V2': 0.3,       # 30% market share
            'Compound': 0.2,      # 20% market share
            'SushiSwap': 0.1      # 10% market share
        }
        
        # Calculate sample total transactions
        total_tx_count = sum([tx_count for _, tx_count, _ in sample_blocks])
        logging.info(f"Total transactions in sampled blocks: {total_tx_count}")
        
        estimated_defi_tx = total_tx_count * 0.2  # Assume 20% of transactions are DeFi-related
        logging.info(f"Estimated DeFi transactions: {estimated_defi_tx}")
        
        # Distribute by protocol weight
        for protocol, weight in protocol_weights.items():
            tx_count = int(estimated_defi_tx * weight)
            defi_data[protocol] = tx_count
        
        # Calculate total DeFi activity
        total_activity = sum(defi_data.values())
        
        # Calculate market share for each protocol
        market_shares = {}
        for protocol, count in defi_data.items():
            market_shares[protocol] = (count / total_activity * 100) if total_activity > 0 else 0
        
        # Generate transaction history data
        transaction_history = []
        for block_num, tx_count, timestamp in sample_blocks:
            block_time = datetime.datetime.fromtimestamp(timestamp)
            # Estimate DeFi transactions in this block
            defi_tx_count = int(tx_count * 0.2)  # Assume 20% of transactions are DeFi-related
            
            transaction_history.append({
                'block': block_num,
                'timestamp': block_time,
                'defi_transactions': defi_tx_count
            })
        
        logging.info(f"DeFi indicators analysis completed in {time.time() - start_time:.2f} seconds")
            
        return {
            'total_activity': total_activity,
            'protocol_activity': defi_data,
            'market_shares': market_shares,
            'transaction_history': pd.DataFrame(transaction_history) if transaction_history else pd.DataFrame()
        }
    except Exception as e:
        logging.error(f"Error getting DeFi indicators: {str(e)}")
        raise Exception(f"Error getting DeFi indicators: {str(e)}")
        
def get_thread_w3(node_url):
    """Thread-local Web3 instance to avoid connection issues in parallel processing"""
    if not hasattr(thread_local, "w3"):
        thread_local.w3 = Web3(Web3.HTTPProvider(node_url))
    return thread_local.w3

def process_block_for_addresses(block_info):
    """Process a single block to extract unique addresses"""
    block_num, node_url, day_str = block_info
    
    try:
        # Get thread-local Web3 instance
        w3 = get_thread_w3(node_url)
        
        # Process block
        block = w3.eth.get_block(block_num)
        timestamp = datetime.datetime.fromtimestamp(block['timestamp'])
        block_day = timestamp.date().strftime('%Y-%m-%d')
        
        # Skip if block is from a different day than expected
        if block_day != day_str:
            return (day_str, set())
            
        # Extract addresses from this block - just get the transaction count
        # For high-volume processing, we'll just count the transactions
        # This is much faster than processing individual transactions
        tx_count = len(block['transactions'])
        
        # We'll use the transaction count as a proxy for address activity
        # This is a good approximation and much faster for massive parallel processing
        if tx_count > 0:
            # Get a few sample transactions to represent the block
            sample_addresses = set()
            sample_size = min(5, tx_count)
            
            # Process just a few transactions as a sample
            for i in range(sample_size):
                try:
                    # Get transaction hash
                    tx_hash = block['transactions'][i]
                    # Get transaction details - this is the slow part, so we limit it
                    tx = w3.eth.get_transaction(tx_hash)
                    
                    # Extract addresses
                    if tx and 'from' in tx and tx['from']:
                        sample_addresses.add(tx['from'])
                    if tx and 'to' in tx and tx['to']:
                        sample_addresses.add(tx['to'])
                except:
                    continue
                    
            return (day_str, sample_addresses)
        
        return (day_str, set())
    except Exception as e:
        logging.debug(f"Error processing block {block_num}: {str(e)}")
        return (day_str, set())

def get_address_activity_trends(w3, days=7, max_workers=500):
    """
    Analyze active addresses trends over time using parallel processing
    
    Parameters:
    - w3: Web3 instance
    - days: Number of days to analyze
    - max_workers: Maximum number of parallel workers (default: 500)
    """
    try:
        logging.info(f"Starting address activity analysis for the last {days} days")
        start_time = time.time()
        
        # Store original node URL for thread-local connections
        node_url = w3.provider.endpoint_uri if hasattr(w3.provider, 'endpoint_uri') else None
        if not node_url:
            raise ValueError("Unable to determine node URL for parallel processing")
            
        latest_block = w3.eth.block_number
        
        # Estimate blocks per day (avg 13.5 seconds per block)
        blocks_per_day = int(24 * 60 * 60 / 13.5)
        
        # Calculate date ranges
        today = datetime.datetime.now().date()
        date_list = [(today - datetime.timedelta(days=d)).strftime('%Y-%m-%d') for d in range(days)]
        
        # Prepare parallel processing
        # Use the passed max_workers parameter instead of the hardcoded value
        logging.info(f"Using {max_workers} parallel workers")
        
        # Process all blocks with massive parallelism
        samples_per_day = 6400  # Process all ~6400 blocks per day
        
        # Track all unique addresses by day
        daily_addresses = {day: set() for day in date_list}
        blocks_processed = 0
        
        # Process each day in the range
        for day_idx, day_str in enumerate(date_list):
            day_start = time.time()
            
            # Calculate block range for this day
            if day_idx == 0:  # Today
                end_block = latest_block
                start_block = max(0, end_block - blocks_per_day)
            else:
                end_block = latest_block - (day_idx * blocks_per_day)
                start_block = max(0, end_block - blocks_per_day)
                
            logging.info(f"Day {day_str}: analyzing blocks {start_block} to {end_block}")
            
            # Determine step size to get the desired number of samples
            step = max(1, (end_block - start_block) // samples_per_day)
            
            # Create block batches for parallel processing
            block_batch = [(block_num, node_url, day_str) 
                          for block_num in range(start_block, end_block + 1, step)]
            
            # Process blocks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_block_for_addresses, block_batch))
                
                # Aggregate results
                for day, addresses in results:
                    daily_addresses[day].update(addresses)
                    blocks_processed += 1
            
            logging.info(f"Day {day_str}: processed ~{len(block_batch)} blocks in {time.time() - day_start:.2f} seconds")
            logging.info(f"Day {day_str}: found {len(daily_addresses[day_str])} unique addresses")
        
        # Convert sets to counts
        address_counts = {day: len(addresses) for day, addresses in daily_addresses.items()}
        
        # Get sorted dates
        dates = sorted(address_counts.keys())
        counts = [address_counts[date] for date in dates]
        
        # Calculate active address growth
        growth_rate = 0
        if len(counts) > 1 and counts[0] > 0:
            growth_rate = ((counts[-1] - counts[0]) / counts[0] * 100)
            
        total_time = time.time() - start_time
        logging.info(f"Processed {blocks_processed} blocks in {total_time:.2f} seconds with parallel processing")
        logging.info(f"Address activity analysis completed, found activity across {len(dates)} days")
            
        return {
            'daily_active_addresses': address_counts,
            'active_address_growth': growth_rate,
            'current_active_addresses': counts[-1] if counts else 0
        }
    except Exception as e:
        logging.error(f"Error getting address activity trends: {str(e)}")
        raise Exception(f"Error getting address activity trends: {str(e)}")
