# Contains token addresses and ABIs for common ERC-20 tokens

# Common ERC-20 tokens with their contract addresses (Ethereum mainnet)
TOKEN_ADDRESSES = {
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    "COMP": "0xc00e94Cb662C3520282E6f5717214004A7f26888",
    "MKR": "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2"
}

# Decimals for common tokens
TOKEN_DECIMALS = {
    "USDC": 6,
    "USDT": 6,
    "DAI": 18,
    "WETH": 18,
    "WBTC": 8,
    "UNI": 18,
    "LINK": 18,
    "AAVE": 18,
    "COMP": 18,
    "MKR": 18
}

# Basic ERC-20 ABI for balanceOf and Transfer events
ERC20_ABI = '''
[
    {
        "constant": true,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "name": "from", "type": "address"},
            {"indexed": true, "name": "to", "type": "address"},
            {"indexed": false, "name": "value", "type": "uint256"}
        ],
        "name": "Transfer",
        "type": "event"
    }
]
'''

# Function to get token info
def get_token_info(token_symbol):
    """Returns token address and decimals for a given symbol"""
    token_symbol = token_symbol.upper()
    if token_symbol in TOKEN_ADDRESSES:
        return {
            "address": TOKEN_ADDRESSES[token_symbol],
            "decimals": TOKEN_DECIMALS[token_symbol]
        }
    return None
