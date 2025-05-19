import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
from blockchain_utils import (
    connect_to_node, 
    get_eth_balance, 
    get_token_balance,
    get_token_transfers,
    get_block_info,
    validate_ethereum_address
)
from token_data import TOKEN_ADDRESSES

# Page configuration
st.set_page_config(
    page_title="On-Chain Data Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if not present
if 'connected' not in st.session_state:
    st.session_state.connected = False

if 'w3' not in st.session_state:
    st.session_state.w3 = None

# Main title
st.title("📊 On-Chain Data Dashboard")
st.markdown("Connect to Coinbase Cloud nodes to analyze blockchain data")

# Sidebar for connection settings
with st.sidebar:
    st.header("Connection Settings")
    
    node_url = st.text_input(
        "Coinbase Cloud Node URL", 
        value=os.getenv("COINBASE_CLOUD_NODE", ""),
        type="password",
        help="Enter your Coinbase Cloud node endpoint URL"
    )
    
    if st.button("Connect to Node"):
        with st.spinner("Connecting to Ethereum node..."):
            try:
                st.session_state.w3 = connect_to_node(node_url)
                st.session_state.connected = True
                st.success("Connected to Ethereum node!")
                
                # Display chain info
                chain_id = st.session_state.w3.eth.chain_id
                chain_name = "Ethereum Mainnet" if chain_id == 1 else f"Chain ID: {chain_id}"
                latest_block = st.session_state.w3.eth.block_number
                
                st.markdown(f"**Network:** {chain_name}")
                st.markdown(f"**Latest Block:** {latest_block}")
            except Exception as e:
                st.error(f"Failed to connect: {str(e)}")
                st.session_state.connected = False
    
    if st.session_state.connected:
        st.success("✅ Connected")
    else:
        st.warning("⚠️ Not connected")

# Only show data dashboard if connected
if st.session_state.connected and st.session_state.w3:
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Wallet Analysis", "Token Transfers", "Block Explorer"])
    
    # Wallet Analysis Tab
    with tab1:
        st.header("Wallet Analysis")
        
        address = st.text_input(
            "Ethereum Address",
            placeholder="0x...",
            help="Enter the Ethereum address to analyze"
        )
        
        tokens_to_check = st.multiselect(
            "Select tokens to analyze",
            options=list(TOKEN_ADDRESSES.keys()),
            default=["USDC", "USDT", "DAI"]
        )
        
        if address and validate_ethereum_address(address):
            with st.spinner("Fetching wallet data..."):
                try:
                    # Get ETH balance
                    eth_balance = get_eth_balance(st.session_state.w3, address)
                    
                    # Create columns for balances
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("ETH Balance", f"{eth_balance:.4f} ETH")
                        
                    # Get token balances
                    token_balances = {}
                    for token in tokens_to_check:
                        try:
                            balance = get_token_balance(st.session_state.w3, address, token)
                            token_balances[token] = balance
                        except Exception as e:
                            st.warning(f"Error fetching {token} balance: {str(e)}")
                    
                    # Display token balances
                    if token_balances:
                        # Create a dataframe for token balances
                        balance_df = pd.DataFrame({
                            'Token': list(token_balances.keys()),
                            'Balance': list(token_balances.values())
                        })
                        
                        # Display balances in a table
                        st.subheader("Token Balances")
                        st.dataframe(balance_df)
                        
                        # Create pie chart for token distribution
                        fig = px.pie(
                            balance_df, 
                            values='Balance', 
                            names='Token', 
                            title='Token Distribution'
                        )
                        st.plotly_chart(fig)
                        
                except Exception as e:
                    st.error(f"Error analyzing wallet: {str(e)}")
        elif address:
            st.error("Invalid Ethereum address format")
    
    # Token Transfers Tab
    with tab2:
        st.header("Token Transfers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            address = st.text_input(
                "Ethereum Address", 
                placeholder="0x...",
                key="transfer_address"
            )
            
        with col2:
            token = st.selectbox(
                "Select Token",
                options=list(TOKEN_ADDRESSES.keys()),
                index=0
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            latest_block = st.session_state.w3.eth.block_number
            from_block = st.number_input(
                "From Block", 
                min_value=0, 
                max_value=latest_block,
                value=latest_block - 5000
            )
            
        with col2:
            to_block = st.number_input(
                "To Block", 
                min_value=from_block,
                max_value=latest_block,
                value=latest_block
            )
        
        if st.button("Analyze Transfers"):
            if address and validate_ethereum_address(address):
                with st.spinner(f"Analyzing {token} transfers..."):
                    try:
                        # Get transfers
                        transfers_data = get_token_transfers(
                            st.session_state.w3, 
                            address, 
                            token, 
                            from_block, 
                            to_block
                        )
                        
                        # Display inflow/outflow metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                f"{token} Inflow", 
                                f"{transfers_data['inflow']:.2f}",
                                delta=None
                            )
                            
                        with col2:
                            st.metric(
                                f"{token} Outflow", 
                                f"{transfers_data['outflow']:.2f}",
                                delta=None
                            )
                            
                        with col3:
                            st.metric(
                                f"Net Flow", 
                                f"{transfers_data['net_flow']:.2f}",
                                delta=None
                            )
                        
                        # Display transfers table
                        if not transfers_data['transfers_df'].empty:
                            st.subheader("Transfer History")
                            st.dataframe(transfers_data['transfers_df'])
                            
                            # Plot transfers over time
                            if len(transfers_data['transfers_df']) > 1:
                                # Create time series for inflows and outflows
                                daily_transfers = transfers_data['transfers_df'].copy()
                                daily_transfers['date'] = daily_transfers['timestamp'].dt.date
                                
                                # Separate inflows and outflows
                                inflows = daily_transfers[daily_transfers['type'] == 'inflow']
                                outflows = daily_transfers[daily_transfers['type'] == 'outflow']
                                
                                # Group by date
                                if not inflows.empty:
                                    inflows_by_date = inflows.groupby('date')['value'].sum().reset_index()
                                else:
                                    inflows_by_date = pd.DataFrame(columns=['date', 'value'])
                                    
                                if not outflows.empty:
                                    outflows_by_date = outflows.groupby('date')['value'].sum().reset_index()
                                else:
                                    outflows_by_date = pd.DataFrame(columns=['date', 'value'])
                                
                                # Create figure
                                fig = go.Figure()
                                
                                if not inflows_by_date.empty:
                                    fig.add_trace(go.Bar(
                                        x=inflows_by_date['date'],
                                        y=inflows_by_date['value'],
                                        name='Inflows',
                                        marker_color='green'
                                    ))
                                
                                if not outflows_by_date.empty:
                                    fig.add_trace(go.Bar(
                                        x=outflows_by_date['date'],
                                        y=outflows_by_date['value'],
                                        name='Outflows',
                                        marker_color='red'
                                    ))
                                
                                fig.update_layout(
                                    title=f'{token} Transfer Activity',
                                    xaxis_title='Date',
                                    yaxis_title=f'{token} Amount',
                                    legend_title='Transaction Type',
                                    barmode='group'
                                )
                                
                                st.plotly_chart(fig)
                        else:
                            st.info(f"No {token} transfers found in the selected block range")
                    except Exception as e:
                        st.error(f"Error analyzing transfers: {str(e)}")
            else:
                st.error("Invalid Ethereum address format")
    
    # Block Explorer Tab
    with tab3:
        st.header("Block Explorer")
        
        latest_block = st.session_state.w3.eth.block_number
        block_number = st.number_input(
            "Block Number", 
            min_value=0,
            max_value=latest_block,
            value=latest_block
        )
        
        if st.button("Get Block Info"):
            with st.spinner("Fetching block data..."):
                try:
                    block_data = get_block_info(st.session_state.w3, block_number)
                    
                    # Display block info
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Block Number", block_data['number'])
                        
                    with col2:
                        st.metric("Timestamp", block_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
                        
                    with col3:
                        st.metric("Transactions", block_data['transactions'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Gas Used", f"{block_data['gas_used']:,}")
                        
                    with col2:
                        gas_percentage = (block_data['gas_used'] / block_data['gas_limit']) * 100
                        st.metric("Gas Limit", f"{block_data['gas_limit']:,}", 
                                 delta=f"{gas_percentage:.2f}% used")
                        
                    # Gas usage visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=gas_percentage,
                        title={'text': "Gas Usage (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 75], 'color': "yellow"},
                                {'range': [75, 100], 'color': "red"}
                            ]
                        }
                    ))
                    
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error fetching block data: {str(e)}")
else:
    # Welcome message when not connected
    st.info("👋 Welcome to the On-Chain Data Dashboard! To get started, connect to a Coinbase Cloud node using the sidebar.")
    
    st.markdown("""
    ## What can you do with this dashboard?
    
    - **Wallet Analysis**: Check ETH and token balances for any Ethereum address
    - **Token Transfers**: Analyze inflows and outflows of tokens for an address
    - **Block Explorer**: View detailed information about specific blocks on the blockchain
    
    ### Getting Started
    
    1. Sign up for a Coinbase Cloud account to get your node endpoint
    2. Enter your node URL in the sidebar and connect
    3. Start exploring on-chain data!
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Web3.py 🚀")
