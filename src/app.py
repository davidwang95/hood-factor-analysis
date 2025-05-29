import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

# Set page config
st.set_page_config(
    page_title="HOOD Factor Analysis",
    page_icon="📈",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    data_dir = Path('data')
    
    # Load price and factor data
    prices = pd.read_csv(data_dir / 'hood_prices.csv', index_col=0, parse_dates=True)
    factors = pd.read_csv(data_dir / 'ff_factors.csv', index_col=0, parse_dates=True)
    
    # Load rolling betas
    betas = pd.read_csv(data_dir / 'rolling_betas.csv', index_col=0, parse_dates=True)
    
    # Load peer comparison
    peers = pd.read_csv(data_dir / 'peer_style_table.csv')
    
    return prices, factors, betas, peers

# Load all data
prices, factors, betas, peers = load_data()

# Sidebar
st.sidebar.title("Settings")
lookback = st.sidebar.slider(
    "Lookback Window (days)",
    min_value=60,
    max_value=len(prices),
    value=252,
    step=30
)

# Ensure all indices are tz-naive
prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)
factors.index = pd.to_datetime(factors.index, utc=True).tz_convert(None)
betas.index = pd.to_datetime(betas.index, utc=True).tz_convert(None)

# Filter data based on lookback
start_date = prices.index[-lookback]
if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
    start_date = pd.Timestamp(start_date).tz_convert(None)
prices = prices.loc[start_date:]
factors = factors.loc[start_date:]
betas = betas.loc[start_date:]

# Add beginner-friendly explanations and definitions
with st.expander("ℹ️ Factor & Metric Definitions (click to expand)", expanded=False):
    st.markdown("""
    **What is the Fama-French Model?**
    The Fama-French model is a widely used asset pricing model developed by Nobel laureate Eugene Fama and Kenneth French. It extends the traditional Capital Asset Pricing Model (CAPM) by including additional factors that help explain stock returns. The model suggests that a stock's return can be explained by its exposure to:
    1. The overall market (like the S&P 500)
    2. Company size (small vs. large companies)
    3. Value vs. growth characteristics
    4. Profitability
    5. Investment patterns
    
    This model helps investors understand what drives a stock's performance and how it compares to the broader market.
    
    **Factor Definitions:**
    - **Mkt-RF**: Market excess return (market return minus risk-free rate)
    - **SMB**: "Small Minus Big" – the size factor (small cap minus large cap returns)
    - **HML**: "High Minus Low" – the value factor (value stocks minus growth stocks)
    - **RMW**: "Robust Minus Weak" – the profitability factor (profitable firms minus unprofitable)
    - **CMA**: "Conservative Minus Aggressive" – the investment factor (conservative investment firms minus aggressive)
    - **RF**: Risk-free rate (e.g., 3-month T-bill rate)
    - **Alpha**: The regression intercept, representing unexplained return
    
    **Peer Metrics:**
    - **Revenue CAGR**: Compound annual growth rate of revenue (trailing 12 months)
    - **P/S Ratio**: Price-to-Sales ratio (valuation metric)
    - **ROE**: Return on Equity (profitability metric)
    - **Beta**: Sensitivity to market movements (systematic risk)
    """)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Price & Factors", "Rolling Betas", "Attribution"])

# Tab 1: Price & Factors
with tab1:
    st.header("Price and Factor Returns")
    st.markdown("""
    **What you see:**
    - The top chart shows the daily closing price of HOOD.
    - The bottom chart shows daily returns for each Fama-French factor and the risk-free rate.
    - See the [Factor & Metric Definitions](#) above for explanations.
    """)
    
    # Create subplot for price and factors
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("HOOD Price", "Factor Returns"),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # Add price plot
    fig.add_trace(
        go.Scatter(x=prices.index, y=prices['price'], name="Price"),
        row=1, col=1
    )
    
    # Add factor plots
    for factor in factors.columns:
        fig.add_trace(
            go.Scatter(x=factors.index, y=factors[factor], name=factor),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Rolling Betas
with tab2:
    st.header("Rolling Factor Betas")
    st.markdown("""
    **What you see:**
    - Each line shows the 60-day rolling sensitivity (beta) of HOOD's excess return to each factor.
    - Higher beta means more exposure to that factor.
    - See the [Factor & Metric Definitions](#) above for explanations.
    """)
    
    # Create plot for rolling betas
    fig = go.Figure()
    
    for col in betas.columns:
        fig.add_trace(
            go.Scatter(x=betas.index, y=betas[col], name=col)
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show latest betas
    st.subheader("Latest Factor Betas")
    st.dataframe(betas.iloc[-1:].T)

# Tab 3: Attribution
with tab3:
    st.header("Factor Attribution")
    st.markdown("""
    **What you see:**
    - The stacked area chart shows how much each factor contributed to HOOD's cumulative return over time.
    - The table below shows the final cumulative contribution of each factor.
    - See the [Factor & Metric Definitions](#) above for explanations.
    """)
    
    # Calculate factor contributions
    contributions = pd.DataFrame(index=factors.index)
    
    for factor in factors.columns:
        beta_col = f'beta_{factor}'
        if beta_col in betas.columns:
            contributions[factor] = betas[beta_col] * factors[factor]
    
    # Add alpha contribution
    contributions['alpha'] = betas['alpha']
    
    # Calculate cumulative contributions
    cumulative = (1 + contributions).cumprod() - 1
    
    # Create stacked area plot
    fig = go.Figure()
    
    for col in cumulative.columns:
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative[col],
                name=col,
                stackgroup='one'
            )
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show final contributions
    st.subheader("Final Cumulative Contributions")
    final_contributions = cumulative.iloc[-1]
    st.dataframe(final_contributions.to_frame('Contribution'))

# Add peer comparison section
st.sidebar.markdown("---")
st.sidebar.header("Peer Comparison")
st.sidebar.dataframe(peers) 