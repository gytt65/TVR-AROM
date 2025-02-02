import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# Page configuration
st.set_page_config(
    page_title="AI-Powered American Option Pricing",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .profit {color: green; font-weight: bold;}
    .loss {color: red; font-weight: bold;}
    .metric-container {
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        text-align: center;
    }
    .metric-call {
        background: linear-gradient(145deg, #90ee90, #76d576);
        color: #1a3c1a;
    }
    .metric-put {
        background: linear-gradient(145deg, #ffcccb, #ffb3b3);
        color: #4a1a1a;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .simulation-progress {
        color: #4a90e2;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit interface
st.title("📊 Advanced Option Pricing Model")

# Sidebar configuration
with st.sidebar:
    st.markdown("<h1 style='font-size: 40px; font-weight: bold;'>📊 Option Pricing Models</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style="font-size: 17px; font-weight: bold; display: inline;">Created by:</p>
    <p style="font-size: 20px; font-weight: bold; display: inline;"> C.A Aniketh<a href="https://www.linkedin.com/in/ca-aniketh-313729225" target="_blank" style="color: inherit; text-decoration: none;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" style="vertical-align: middle;">
        </a>
    </p>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Model parameters
    st.title("⚙️ Pricing Parameters")
    S0 = st.number_input("Current Price (₹)", value=23482.15, step=100.0)
    K = st.number_input("Strike Price (₹)", value=24000.0, step=100.0)
    days_to_maturity = st.number_input("Time to Maturity (Days)", value=5, min_value=1, step=1)
    T = days_to_maturity / 360  # Convert to years
    r = st.number_input("Risk-Free Rate (%)", value=6.9, step=0.1) / 100
    sigma = st.number_input("Volatility (%)", value=14.09, step=0.1) / 100
    call_purchase = st.number_input("Call Purchase Price (₹)", value=0.0)
    put_purchase = st.number_input("Put Purchase Price (₹)", value=0.0)
    
    st.markdown("---")
    st.title("Model Configuration")
    N = st.selectbox("Simulations", [10000, 50000, 100000], index=2)
    M = st.selectbox("Time Steps", [50, 100, 200], index=1)
    degree = st.slider("Polynomial Degree", 2, 5, 3)
    alpha = st.slider("Regularization (α)", 0.0, 2.0, 0.5, step=0.1)
    seed = st.number_input("Random Seed", value=42)

def generate_asset_paths(S0, r, sigma, T, M, N, seed=None):
    np.random.seed(seed)
    if N % 2 != 0:
        N += 1
    dt = T / M
    half_N = N // 2
    S = np.zeros((N, M + 1))
    S[:, 0] = S0
    for t in range(1, M + 1):
        Z = np.random.standard_normal(half_N)
        Z = np.concatenate([Z, -Z])
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return S, dt
# Add this function to generate fake candlestick data
def generate_fake_candles(num=20, initial_price=100):
    np.random.seed()
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num, freq='15min')
    prices = []
    current_price = initial_price
    
    for _ in range(num):
        movement = np.random.choice([0.98, 0.99, 1.0, 1.01, 1.02])
        current_price *= movement + np.random.normal(0, 0.005)
        prices.append(current_price)
    
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    df['Open'] = df['Close'].shift(1).fillna(initial_price)
    df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.0, 1.02, num)
    df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.98, 1.0, num)
    df['Close'] = df['Close'] * np.random.uniform(0.99, 1.01, num)
    return df


def american_option_pricing(S0, K, T, r, sigma, option_type='put', 
                           N=100000, M=100, degree=3, alpha=1.0, seed=None):
    S, dt = generate_asset_paths(S0, r, sigma, T, M, N, seed)
    
    if option_type == 'put':
        payoff = np.maximum(K - S[:, -1], 0)
        itm_condition = lambda S: S < K
        exercise_value = lambda S: K - S
    elif option_type == 'call':
        payoff = np.maximum(S[:, -1] - K, 0)
        itm_condition = lambda S: S > K
        exercise_value = lambda S: S - K
        
    cash_flows = payoff * np.exp(-r * T)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Initialize visualization elements
    chart_placeholder = st.empty()
    
    for t in range(M-1, 0, -1):
        current_time = t * dt
        time_remaining = T - current_time
        in_the_money = itm_condition(S[:, t])
        
        if not in_the_money.any():
            continue
            
        X_in = S[in_the_money, t]
        y_in = cash_flows[in_the_money]
        X_time = np.full_like(X_in, time_remaining)
        X_features = np.column_stack((X_in, X_time))
        X_poly = poly.fit_transform(X_features)
        
        weights = exercise_value(X_in) if option_type == 'put' else X_in - K
        weights = np.abs(weights).clip(min=1e-6)
        
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(X_poly, y_in, sample_weight=weights)
        continuation_est = model.predict(X_poly)
        
        immediate_PV = exercise_value(X_in) * np.exp(-r * current_time)
        exercise_mask = immediate_PV > continuation_est
        
        cash_flows[in_the_money] = np.where(exercise_mask, immediate_PV, y_in)
        
        # Update candlestick visualization every 5 steps
        if t % 5 == 0:
            df = generate_fake_candles(num=20, initial_price=S0)
            fig = go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            fig.update_layout(
                title='Live Market Simulation',
                xaxis_title='Time',
                yaxis_title='Price (₹)',
                template='plotly_white',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Clear visualization elements after completion
    chart_placeholder.empty()
    
    return np.mean(cash_flows), np.std(cash_flows) / np.sqrt(N)

# Corrected Early Exercise Boundary
def plot_early_exercise_boundary(S0, K, T, r, sigma):
    time_steps = np.linspace(0, T, 20)  # Reduced steps for faster computation
    boundaries = []
    
    for t in time_steps:
        # Corrected boundary estimation
        price, _ = american_option_pricing(S0, K, max(t, 0.001), r, sigma, 'put', 10000, 50)
        boundaries.append(K - price)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_steps * 365,
        y=boundaries,
        mode='lines+markers',
        name='Exercise Boundary',
        line=dict(color='#FF6F00')
    ))
    fig.update_layout(
        title='Early Exercise Boundary Over Time',
        xaxis_title='Days to Expiry',
        yaxis_title='Critical Spot Price (₹)',
        hovermode="x unified",
        template='plotly_white'
    )
    return fig


# Corrected Greeks Calculation
def calculate_greeks(S0, K, T, r, sigma):
    dS = S0 * 0.01  # 1% perturbation
    dSigma = sigma * 0.01  # 1% volatility change
    
    # Delta calculation
    price_up, _ = american_option_pricing(S0 + dS, K, T, r, sigma, 'put')
    price_down, _ = american_option_pricing(S0 - dS, K, T, r, sigma, 'put')
    delta = (price_up - price_down) / (2 * dS)
    
    # Vega calculation
    price_vol_up, _ = american_option_pricing(S0, K, T, r, sigma + dSigma, 'put')
    price_vol_down, _ = american_option_pricing(S0, K, T, r, sigma - dSigma, 'put')
    vega = (price_vol_up - price_vol_down) / (2 * dSigma)
    
    return {'Delta': delta, 'Vega': vega}

# Main interface
st.markdown("### Advanced Pricing Model with P&L Analysis")

# Corrected Scenario Comparison
def run_scenario_analysis(S0, K, T, r, sigma):
    scenarios = {
        'Bull Market': {'spot': S0 * 1.2, 'vol': sigma * 0.8},
        'Bear Market': {'spot': S0 * 0.8, 'vol': sigma * 1.2},
        'Volatility Spike': {'spot': S0, 'vol': sigma * 1.5}
    }
    
    results = []
    for name, params in scenarios.items():
        price, _ = american_option_pricing(params['spot'], K, T, r, params['vol'], 'put')
        results.append({
            'Scenario': name,
            'Spot Price': params['spot'],
            'Volatility': f"{params['vol'] * 100:.1f}%",
            'Option Price': price
        })
    
    return pd.DataFrame(results)

# Price and P&L display
col1, col2 = st.columns(2)
with col1:
    with st.spinner("Calculating CALL option..."):
        call_price, call_se = american_option_pricing(S0, K, T, r, sigma, 'call', N, M, degree, alpha, seed)
    call_pnl = call_price - call_purchase
    pnl_class = "profit" if call_pnl >= 0 else "loss"
    
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div class="metric-label">American CALL Value</div>
            <div class="metric-value">₹{call_price:,.2f}</div>
            <div>± {call_se:.4f} (SE)</div>
            <div style="margin-top: 15px;">
                P&L: <span class="{pnl_class}">₹{call_pnl:,.2f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    with st.spinner("Calculating PUT option..."):
        put_price, put_se = american_option_pricing(S0, K, T, r, sigma, 'put', N, M, degree, alpha, seed)
    put_pnl = put_price - put_purchase
    pnl_class = "profit" if put_pnl >= 0 else "loss"
    
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div class="metric-label">American PUT Value</div>
            <div class="metric-value">₹{put_price:,.2f}</div>
            <div>± {put_se:.4f} (SE)</div>
            <div style="margin-top: 15px;">
                P&L: <span class="{pnl_class}">₹{put_pnl:,.2f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)



# Updated Main Interface
st.markdown("---")
st.title("📉 Greeks Analysis")
greeks = calculate_greeks(S0, K, T, r, sigma)

col1, col2 = st.columns(2)
with col1:
    st.metric("Delta", value=f"{greeks['Delta']:.4f}", 
              help="Price sensitivity to underlying asset price changes")

with col2:
    st.metric("Vega", value=f"{greeks['Vega']:.4f}", 
              help="Price sensitivity to volatility changes")

# Updated Early Exercise Boundary
st.markdown("---")
st.title("⚡ Early Exercise Boundary")
boundary_fig = plot_early_exercise_boundary(S0, K, T, r, sigma)
st.plotly_chart(boundary_fig, use_container_width=True)

# Updated Scenario Analysis
st.markdown("---")
st.title("📚 Scenario Comparison")
df_scenarios = run_scenario_analysis(S0, K, T, r, sigma)
st.dataframe(
    df_scenarios.style.format({
        'Spot Price': '₹{:.2f}',
        'Option Price': '₹{:.2f}'
    }),
    height=150,
    use_container_width=True
)

# Monte Carlo Path Visualization
st.markdown("---")
st.title("🌐 Simulation Path Explorer")
if st.button("Generate New Paths"):
    S, _ = generate_asset_paths(S0, r, sigma, T, M, 50, seed)
    fig = go.Figure()
    for i in range(S.shape[0]):
        fig.add_trace(go.Scatter(
            x=np.linspace(0, days_to_maturity, M+1),
            y=S[i],
            mode='lines',
            line=dict(width=1),
            showlegend=False
        ))
    fig.update_layout(
        title='Monte Carlo Simulation Paths',
        xaxis_title='Days to Expiry',
        yaxis_title='Spot Price (₹)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
