import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

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
st.title("📊 American Option Pricing Model")

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
    T = days_to_maturity / 365  # Fixed to 365-day year
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
    N = (N // 2) * 2  # Ensure even number
    dt = T / M
    Z = np.random.standard_normal((N//2, M))
    Z = np.concatenate([Z, -Z])  # Proper antithetic variates
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=1)
    S = S0 * np.exp(log_paths)
    return np.insert(S, 0, S0, axis=1), dt

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
    
    chart_placeholder = st.empty()
    
    for t in range(M-1, 0, -1):
        current_time = t * dt
        time_remaining = T - current_time
        in_the_money = itm_condition(S[:, t])
        
        if not in_the_money.any():
            continue
            
        X_in = S[in_the_money, t]
        y_in = cash_flows[in_the_money]
        
        # Feature normalization
        X_normalized = X_in / S0  # Normalize by spot price
        time_normalized = np.full_like(X_normalized, time_remaining / T)  # Match shape
        X_features = np.column_stack((X_normalized, time_normalized))  # Now compatible
        
        X_poly = poly.fit_transform(X_features)
        
        weights = exercise_value(X_in).clip(min=1e-6)
        
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(X_poly, y_in, sample_weight=np.abs(weights))
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
    
    chart_placeholder.empty()
    
    return np.mean(cash_flows), np.std(cash_flows) / np.sqrt(N)
    

@st.cache_data
def plot_early_exercise_boundary(S0, K, T, r, sigma):
    time_steps = np.linspace(0, T, 20)
    boundaries = []
    
    # Precompute paths once
    S, _ = generate_asset_paths(S0, r, sigma, T, 50, 10000)
    
    for t in time_steps:
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

# Update the calculate_greeks function
def calculate_greeks(S0, K, T, r, sigma, option_type='put'):
    """Calculate all Greeks using finite differences for American options"""
    # Perturbation parameters
    dS = S0 * 0.001  # 0.1% of spot price
    dSigma = 0.001    # 0.1% absolute volatility change
    dT = 1/365       # 1 day time decay
    dr = 0.0001      # 0.01% interest rate change
    
    # Base price
    base_price = american_option_pricing(S0, K, T, r, sigma, option_type)[0]
    
    # Delta calculation
    price_up = american_option_pricing(S0 + dS, K, T, r, sigma, option_type)[0]
    price_down = american_option_pricing(S0 - dS, K, T, r, sigma, option_type)[0]
    delta = (price_up - price_down) / (2 * dS)
    
    # Gamma calculation
    gamma = (price_up - 2*base_price + price_down) / (dS ** 2)
    
    # Vega calculation
    price_vol_up = american_option_pricing(S0, K, T, r, sigma + dSigma, option_type)[0]
    price_vol_down = american_option_pricing(S0, K, T, r, sigma - dSigma, option_type)[0]
    vega = (price_vol_up - price_vol_down) / (2 * dSigma)
    
    # Theta calculation (1 day decay)
    new_T = max(T - dT, 1e-5)
    price_T = american_option_pricing(S0, K, new_T, r, sigma, option_type)[0]
    theta = (price_T - base_price) / dT
    
    # Rho calculation
    price_r_up = american_option_pricing(S0, K, T, r + dr, sigma, option_type)[0]
    price_r_down = american_option_pricing(S0, K, T, r - dr, sigma, option_type)[0]
    rho = (price_r_up - price_r_down) / (2 * dr)
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }

# Main interface
st.markdown("### American Option Pricing Model with P&L Analysis")

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

st.markdown("---")
st.title("📉 Greeks Analysis")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option Greeks")
    greeks_call = calculate_greeks(S0, K, T, r, sigma, 'call')
    st.metric("Delta", f"{greeks_call['Delta']:.4f}", 
             help="Sensitivity to underlying price changes")
    st.metric("Gamma", f"{greeks_call['Gamma']:.4f}", 
             help="Sensitivity to delta changes")
    st.metric("Vega", f"{greeks_call['Vega']:.4f}", 
             help="Sensitivity to volatility changes (per 1% change)")
    st.metric("Theta", f"{greeks_call['Theta']:.4f}", 
             help="Daily time decay (1 day)")
    st.metric("Rho", f"{greeks_call['Rho']:.4f}", 
             help="Sensitivity to interest rate changes (per 1% change)")

with col2:
    st.subheader("Put Option Greeks")
    greeks_put = calculate_greeks(S0, K, T, r, sigma, 'put')
    st.metric("Delta", f"{greeks_put['Delta']:.4f}", 
             help="Sensitivity to underlying price changes")
    st.metric("Gamma", f"{greeks_put['Gamma']:.4f}", 
             help="Sensitivity to delta changes")
    st.metric("Vega", f"{greeks_put['Vega']:.4f}", 
             help="Sensitivity to volatility changes (per 1% change)")
    st.metric("Theta", f"{greeks_put['Theta']:.4f}", 
             help="Daily time decay (1 day)")
    st.metric("Rho", f"{greeks_put['Rho']:.4f}", 
             help="Sensitivity to interest rate changes (per 1% change)")

st.markdown("---")
st.title("⚡ Early Exercise Boundary")
boundary_fig = plot_early_exercise_boundary(S0, K, T, r, sigma)
st.plotly_chart(boundary_fig, use_container_width=True)

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
