import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import numpy as np
import logging
from blockchain_utils import (
    connect_to_node, 
    get_network_metrics,
    get_defi_indicators,
    get_address_activity_trends
)
# We no longer need these imports since we removed those sections
# from token_data import TOKEN_ADDRESSES

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
                logging.info(f"Attempting to connect to node with URL: {node_url[:10]}[...]")
                
                if not node_url or node_url.strip() == "":
                    raise ValueError("Node URL is empty. Please enter your Coinbase Cloud node URL.")
                
                st.session_state.w3 = connect_to_node(node_url)
                
                # Check connection
                if not st.session_state.w3.is_connected():
                    raise ConnectionError("Web3 instance created but is_connected() returned False")
                    
                st.session_state.connected = True
                st.success("Connected to Ethereum node!")
                
                # Display chain info
                chain_id = st.session_state.w3.eth.chain_id
                chain_name = "Ethereum Mainnet" if chain_id == 1 else f"Chain ID: {chain_id}"
                latest_block = st.session_state.w3.eth.block_number
                
                logging.info(f"Successfully connected to {chain_name} - Latest block: {latest_block}")
                
                st.markdown(f"**Network:** {chain_name}")
                st.markdown(f"**Latest Block:** {latest_block}")
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Connection error: {error_msg}")
                
                # More descriptive error message
                if "Failed to connect" in error_msg:
                    error_msg = "Failed to connect to the Ethereum node. Please check your node URL and ensure it's accessible."
                elif "timeout" in error_msg.lower():
                    error_msg = "Connection timed out. The node might be experiencing high load or the URL might be incorrect."
                elif "Invalid provider" in error_msg:
                    error_msg = "Invalid provider URL. Please ensure the URL is properly formatted and includes the protocol (https://)."
                
                st.error(f"Failed to connect: {error_msg}")
                st.session_state.connected = False
    
    if st.session_state.connected:
        st.success("✅ Connected")
    else:
        st.warning("⚠️ Not connected")

# Only show data dashboard if connected
if st.session_state.connected and st.session_state.w3:
    # Only show Market Indicators tab
    tab1 = st.tabs(["Market Indicators"])[0]
    
    # Market Indicators Tab - Forward Looking Analysis
    with tab1:
        st.header("Market Indicators")
        st.subheader("On-Chain Forward-Looking Indicators")
        
        st.markdown("""
        This section provides network-wide metrics that can help identify trends before they appear in price movements.
        These indicators analyze blockchain behavior that often precedes market changes.
        """)
        
        analysis_type = st.radio(
            "Select Analysis Type",
            options=["Network Activity", "DeFi Activity", "Address Growth"]
        )
        
        if analysis_type == "Network Activity":
            st.subheader("Network Activity Metrics")
            
            # Settings for analysis
            col1, col2 = st.columns(2)
            with col1:
                blocks_back = st.slider("Blocks to analyze", 50, 500, 100)
            
            if st.button("Analyze Network Activity"):
                with st.spinner("Analyzing network metrics... (this may take a moment)"):
                    try:
                        # Get network metrics
                        metrics = get_network_metrics(st.session_state.w3, blocks_back)
                        
                        # Display key metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Transaction Growth Rate", 
                                f"{metrics['tx_growth_rate']:.2f}%",
                                delta=f"{metrics['tx_growth_rate']:.1f}%" if metrics['tx_growth_rate'] != 0 else None
                            )
                            
                            st.metric(
                                "Average Gas Price", 
                                f"{metrics['avg_gas_price']:.2f} Gwei"
                            )
                        
                        with col2:
                            network_health = "Healthy" if metrics['avg_gas_utilization'] < 80 else "Congested"
                            st.metric(
                                "Network Congestion", 
                                f"{metrics['avg_gas_utilization']:.2f}%",
                                delta=network_health
                            )
                            
                            st.metric(
                                "Average Block Time", 
                                f"{metrics['avg_block_time']:.2f} seconds"
                            )
                        
                        # Chart for transaction counts over time
                        if not metrics['block_data'].empty:
                            st.subheader("Transaction Activity Over Time")
                            
                            # Plot transaction count trend
                            fig = px.scatter(
                                metrics['block_data'], 
                                x='timestamp', 
                                y='transaction_count',
                                trendline='lowess',
                                title='Transaction Volume Trend'
                            )
                            
                            st.plotly_chart(fig)
                            
                            # Gas utilization over time
                            st.subheader("Network Utilization")
                            fig = px.line(
                                metrics['block_data'],
                                x='timestamp',
                                y='gas_utilization',
                                title='Gas Utilization Over Time'
                            )
                            
                            # Add a horizontal line at 80% utilization (congestion threshold)
                            fig.add_hline(
                                y=0.8, 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text="Congestion Threshold"
                            )
                            
                            st.plotly_chart(fig)
                            
                            # Add interpretations based on the metrics
                            st.subheader("Indicator Interpretations")
                            
                            # Transaction growth interpretation
                            if metrics['tx_growth_rate'] > 20:
                                st.success("✅ Strong transaction growth indicates increasing network adoption and potential price appreciation")
                            elif metrics['tx_growth_rate'] > 5:
                                st.info("ℹ️ Moderate transaction growth suggests steady network activity")
                            elif metrics['tx_growth_rate'] < -10:
                                st.warning("⚠️ Declining transaction activity may indicate reduced network usage")
                                
                            # Gas price interpretation
                            if metrics['avg_gas_price'] > 50:
                                st.warning("⚠️ High gas prices indicate network congestion and high demand")
                            elif metrics['avg_gas_price'] < 20:
                                st.success("✅ Low gas prices suggest the network is operating efficiently")
                                
                            # Network congestion interpretation
                            if metrics['avg_gas_utilization'] > 80:
                                st.warning("⚠️ Network is highly congested, which may impact transaction speeds")
                            elif metrics['avg_gas_utilization'] < 50:
                                st.success("✅ Network has ample capacity for growth")
                                
                            # Download data button
                            st.subheader("Download Raw Data")
                            
                            # Download block data as CSV
                            csv = metrics['block_data'].to_csv(index=False)
                            st.download_button(
                                label="Download Network Data (CSV)",
                                data=csv,
                                file_name="network_metrics.csv",
                                mime="text/csv",
                            )
                    except Exception as e:
                        st.error(f"Error analyzing network metrics: {str(e)}")
        
        elif analysis_type == "DeFi Activity":
            st.subheader("DeFi Activity Indicators")
            
            col1, col2 = st.columns(2)
            with col1:
                blocks_back = st.slider("Blocks to analyze", 100, 5000, 1000)
            
            if st.button("Analyze DeFi Activity"):
                with st.spinner("Analyzing DeFi metrics... (this may take a moment)"):
                    try:
                        # Get DeFi indicators
                        defi_data = get_defi_indicators(st.session_state.w3, blocks_back)
                        
                        # Display total activity
                        st.metric(
                            "Total DeFi Activity", 
                            f"{defi_data['total_activity']} transactions"
                        )
                        
                        # Create protocol activity chart
                        if defi_data['protocol_activity']:
                            # Convert to dataframe for charting
                            df = pd.DataFrame({
                                'Protocol': list(defi_data['protocol_activity'].keys()),
                                'Activity': list(defi_data['protocol_activity'].values()),
                                'Market Share (%)': [defi_data['market_shares'][p] for p in defi_data['protocol_activity'].keys()]
                            })
                            
                            # Protocol activity chart
                            fig = px.bar(
                                df,
                                x='Protocol',
                                y='Activity',
                                title='DeFi Protocol Activity',
                                color='Protocol'
                            )
                            
                            st.plotly_chart(fig)
                            
                            # Market share pie chart
                            fig = px.pie(
                                df,
                                values='Market Share (%)',
                                names='Protocol',
                                title='DeFi Protocol Market Share'
                            )
                            
                            st.plotly_chart(fig)
                            
                            # Interpretations based on the data
                            st.subheader("Indicator Interpretations")
                            
                            # Find the most active protocol
                            most_active = df.loc[df['Activity'].idxmax()]
                            
                            st.info(f"ℹ️ {most_active['Protocol']} is currently the most active DeFi protocol with {most_active['Market Share (%)']:.2f}% market share")
                            
                            # Calculate concentration (higher means more dominance by one protocol)
                            herfindahl_index = sum((ms/100)**2 for ms in df['Market Share (%)'])
                            
                            if herfindahl_index > 0.25:
                                st.warning("⚠️ DeFi activity is concentrated in fewer protocols, indicating potential centralization risks")
                            else:
                                st.success("✅ DeFi activity is well distributed across protocols, indicating a healthy ecosystem")
                                
                            # Add download data button for DeFi data
                            st.subheader("Download Raw Data")
                            
                            # Download protocol activity data
                            csv_protocols = df.to_csv(index=False)
                            st.download_button(
                                label="Download Protocol Data (CSV)",
                                data=csv_protocols,
                                file_name="defi_protocol_data.csv",
                                mime="text/csv",
                            )
                            
                            # If we have transaction history, add a download button for that too
                            if not defi_data['transaction_history'].empty:
                                csv_history = defi_data['transaction_history'].to_csv(index=False)
                                st.download_button(
                                    label="Download DeFi Transaction History (CSV)",
                                    data=csv_history,
                                    file_name="defi_transaction_history.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.info("No DeFi activity detected in the selected block range")
                    except Exception as e:
                        st.error(f"Error analyzing DeFi metrics: {str(e)}")
        
        elif analysis_type == "Address Growth":
            st.subheader("Active Address Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                days = st.slider("Days to analyze", 1, 14, 7)
            
            if st.button("Analyze Address Activity"):
                with st.spinner("Analyzing address activity... (this may take a while)"):
                    try:
                        # Get address activity trends
                        address_data = get_address_activity_trends(st.session_state.w3, days)
                        
                        # Display current active addresses
                        st.metric(
                            "Current Active Addresses", 
                            f"{address_data['current_active_addresses']:,}",
                            delta=f"{address_data['active_address_growth']:.2f}% growth" if address_data['active_address_growth'] != 0 else None
                        )
                        
                        # Create chart of active addresses over time
                        if address_data['daily_active_addresses']:
                            # Convert to dataframe for charting
                            dates = list(address_data['daily_active_addresses'].keys())
                            counts = list(address_data['daily_active_addresses'].values())
                            
                            df = pd.DataFrame({
                                'Date': dates,
                                'Active Addresses': counts
                            })
                            
                            # Line chart of active addresses
                            fig = px.line(
                                df,
                                x='Date',
                                y='Active Addresses',
                                title='Daily Active Addresses',
                                markers=True
                            )
                            
                            st.plotly_chart(fig)
                            
                            # Interpretations
                            st.subheader("Indicator Interpretations")
                            
                            if address_data['active_address_growth'] > 10:
                                st.success("✅ Strong growth in active addresses suggests increasing network adoption")
                            elif address_data['active_address_growth'] > 0:
                                st.info("ℹ️ Moderate growth in active addresses indicates steady network usage")
                            else:
                                st.warning("⚠️ Decline in active addresses may indicate reduced network participation")
                                
                            # Calculate trend direction (are more recent days higher or lower?)
                            if len(counts) > 1:
                                first_half_avg = np.mean(counts[:len(counts)//2])
                                second_half_avg = np.mean(counts[len(counts)//2:])
                                
                                if second_half_avg > first_half_avg * 1.05:
                                    st.success("✅ Accelerating user growth - a leading indicator of potential ecosystem expansion")
                                elif second_half_avg < first_half_avg * 0.95:
                                    st.warning("⚠️ Recent decrease in active users - monitor for potential network activity slowdown")
                                    
                            # Add download data button for address activity data
                            st.subheader("Download Raw Data")
                            
                            # Download address activity data
                            csv_addresses = df.to_csv(index=False)
                            st.download_button(
                                label="Download Address Activity Data (CSV)",
                                data=csv_addresses,
                                file_name="address_activity_data.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info("Insufficient address activity data for the selected time period")
                    except Exception as e:
                        st.error(f"Error analyzing address metrics: {str(e)}")
        
        # Documentation
        with st.expander("Understanding Forward-Looking Indicators"):
            st.markdown("""
            ### Network Activity
            - **Transaction Growth Rate**: Increasing transaction volume often precedes price movements
            - **Gas Prices**: Elevated gas prices indicate high demand for block space
            - **Network Congestion**: High utilization suggests strong demand and potential scaling challenges
            
            ### DeFi Activity
            - **Total Activity**: Measures overall DeFi ecosystem engagement
            - **Protocol Distribution**: A healthy ecosystem has activity distributed across multiple protocols
            - **Market Concentration**: Low concentration suggests reduced systemic risk
            
            ### Address Growth
            - **Active Addresses**: Growing number of active addresses indicates network adoption
            - **Growth Rate**: Acceleration in new address creation often precedes bullish price movements
            - **Activity Patterns**: Changes in user behavior can signal market shifts
            """)
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
