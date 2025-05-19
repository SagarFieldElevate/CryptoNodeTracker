import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import numpy as np
import logging
import json
from blockchain_utils import (
    connect_to_node, 
    get_network_metrics,
    get_defi_indicators,
    get_address_activity_trends
)
from ai_insights import generate_blockchain_insights
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
            options=["Network Activity", "DeFi Activity", "Address Growth", "AI Insights"]
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
                        
        elif analysis_type == "AI Insights":
            st.subheader("AI-Powered Blockchain Insights")
            
            # Check if we have an OpenAI API key
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                st.warning("⚠️ OpenAI API key is required for AI-powered insights. Please add your API key to continue.")
                
                # Add API key input
                api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI insights")
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.success("API key set. You can now generate AI insights.")
                    st.rerun()
            
            # Analysis settings
            st.markdown("### Analysis Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                network_blocks = st.slider("Network Blocks to Analyze", 50, 500, 200)
            with col2:
                address_days = st.slider("Address History Days", 1, 14, 3)
            
            # Button to generate insights
            if st.button("Generate AI Insights"):
                with st.spinner("Gathering blockchain data and generating AI insights..."):
                    try:
                        # First collect all the blockchain data from different sources
                        chain_id = st.session_state.w3.eth.chain_id
                        
                        # 1. Get network metrics
                        network_metrics = get_network_metrics(st.session_state.w3, network_blocks)
                        
                        # 2. Get DeFi indicators
                        defi_metrics = get_defi_indicators(st.session_state.w3, network_blocks * 10)
                        
                        # 3. Get address activity trends
                        address_metrics = get_address_activity_trends(st.session_state.w3, address_days)
                        
                        # Compile all data
                        blockchain_data = {
                            'timestamp': datetime.datetime.now().isoformat(),
                            'network_metrics': network_metrics,
                            'defi_metrics': defi_metrics,
                            'address_metrics': address_metrics
                        }
                        
                        # Generate insights using AI
                        st.markdown("### Analyzing On-Chain Data")
                        st.markdown("Generating AI-powered insights based on collected blockchain data...")
                        
                        insights_result = generate_blockchain_insights(blockchain_data, chain_id)
                        
                        if 'error' in insights_result:
                            st.error(insights_result['error'])
                            st.markdown(insights_result.get('message', ''))
                        else:
                            # Display the insights
                            st.markdown("## AI-Generated Blockchain Insights")
                            
                            # Main summary
                            if 'summary' in insights_result:
                                st.markdown(f"### Summary\n{insights_result['summary']}")
                            
                            # Market trends
                            if 'market_trends' in insights_result:
                                st.markdown("### Market Trends")
                                market_trends = insights_result['market_trends']
                                
                                # Handle both list of strings and list of dictionaries
                                if isinstance(market_trends, list):
                                    for trend in market_trends:
                                        if isinstance(trend, dict) and 'sentiment' in trend:
                                            if trend.get('sentiment') == 'positive':
                                                st.success(f"✅ {trend.get('insight', '')}")
                                            elif trend.get('sentiment') == 'negative':
                                                st.warning(f"⚠️ {trend.get('insight', '')}")
                                            else:
                                                st.info(f"ℹ️ {trend.get('insight', '')}")
                                        else:
                                            st.markdown(f"- {trend}")
                                else:
                                    st.markdown(market_trends)
                            
                            # Network health
                            if 'network_health' in insights_result:
                                st.markdown("### Network Health")
                                if isinstance(insights_result['network_health'], list):
                                    for item in insights_result['network_health']:
                                        st.markdown(f"- {item}")
                                else:
                                    st.markdown(insights_result['network_health'])
                            
                            # Key indicators
                            if 'key_indicators' in insights_result:
                                st.markdown("### Key Indicators to Watch")
                                if isinstance(insights_result['key_indicators'], list):
                                    for item in insights_result['key_indicators']:
                                        st.markdown(f"- {item}")
                                else:
                                    st.markdown(insights_result['key_indicators'])
                            
                            # Recommendations
                            if 'recommendations' in insights_result:
                                st.markdown("### Strategic Recommendations")
                                if isinstance(insights_result['recommendations'], list):
                                    for item in insights_result['recommendations']:
                                        st.markdown(f"- {item}")
                                else:
                                    st.markdown(insights_result['recommendations'])
                            
                            # Raw JSON for developers
                            with st.expander("View Raw AI Analysis"):
                                st.json(insights_result)
                            
                            # Allow downloading the insights
                            insights_json = json.dumps(insights_result, indent=2)
                            st.download_button(
                                label="Download AI Insights (JSON)",
                                data=insights_json,
                                file_name="blockchain_ai_insights.json",
                                mime="application/json",
                            )
                    except Exception as e:
                        st.error(f"Error generating AI insights: {str(e)}")
            
            # Documentation about AI insights
            with st.expander("About AI-Powered Blockchain Insights"):
                st.markdown("""
                ### How It Works
                This feature uses a state-of-the-art Large Language Model (GPT-4o) to analyze on-chain data and generate insights that may help predict future market trends.
                
                ### Data Sources
                - **Network Activity**: Transaction volumes, gas prices, and network congestion
                - **DeFi Activity**: Protocol usage, market concentration, and activity trends
                - **Address Growth**: Active addresses and user growth patterns
                
                ### Insights Provided
                - **Market Trends**: Potential directional movements based on on-chain indicators
                - **Network Health**: Assessment of blockchain scalability and adoption metrics
                - **Strategic Recommendations**: Actionable suggestions based on the data analysis
                
                ### Limitations
                - AI insights are not financial advice
                - Analysis is based on available on-chain data and should be used as one of many inputs for decision-making
                - Past patterns may not predict future results
                """)
        
        
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
