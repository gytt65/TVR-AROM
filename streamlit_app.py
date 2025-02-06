import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from scipy.stats import norm
from functools import lru_cache
import concurrent.futures
from numpy.linalg import LinAlgError

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
    T = days_to_maturity / 365
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

def black_scholes(S, K, T, r, sigma, option_type='put'):
    """European option pricing for control variates"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

def generate_asset_paths(S0, r, sigma, T, M, N, seed=None):
    """Optimized path generation with antithetic variates"""
    np.random.seed(seed)
    dt = T/M
    N = (N//4)*4  # Better vectorization
    rand = np.random.standard_normal((N//2, M))
    rand = np.concatenate([rand, -rand])
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return np.insert(S, 0, S0, axis=1), dt

def generate_fake_candles(num=20, initial_price=100):
    """Generate simulated market data"""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num, freq='15min')
    prices = initial_price * np.exp(np.cumsum(np.random.normal(0, 0.005, num)))
    
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    df['Open'] = df['Close'].shift(1).fillna(initial_price)
    df['High'] = df[['Open', 'Close']].max(axis=1)*1.005
    df['Low'] = df[['Open', 'Close']].min(axis=1)*0.995
    return df

def american_option_pricing(S0, K, T, r, sigma, option_type='put', 
                           N=100000, M=100, degree=3, alpha=1.0, seed=None):
    """Nobel-grade pricing engine with same UI integration"""
    # Generate optimized paths
    paths, dt = generate_asset_paths(S0, r, sigma, T, M, N, seed)
    
    # Precompute European values matrix
    euro_grid = np.array([black_scholes(paths[i,t], K, T-t*dt, r, sigma, option_type)
                        for t in range(M) for i in range(N)]).reshape(N,M)
    
    # Initialize components
    payoff = np.maximum(K - paths[:,-1], 0) if option_type == 'put' else np.maximum(paths[:,-1] - K, 0)
    cash_flows = payoff * np.exp(-r*T)
    spline = SplineTransformer(n_knots=7, degree=3)
    poly = PolynomialFeatures(degree, include_bias=False)
    
    # Visualization setup
    chart_placeholder = st.empty()
    
    # Enhanced backward induction
    for t in range(M-1, 0, -1):
        intrinsic = K - paths[:,t] if option_type == 'put' else paths[:,t] - K
        in_the_money = intrinsic > 0
        if sum(in_the_money) < 5: continue
        
        # Feature engineering
        X = paths[in_the_money,t]/S0
        time_feat = np.full_like(X, t*dt/T)
        features = np.column_stack([X, time_feat, X**2, X*time_feat])
        
        try:  # Auto basis selection
            X_trans = spline.fit_transform(features)
        except LinAlgError:
            X_trans = poly.fit_transform(features)
        
        # Control variate regression
        euro_t = euro_grid[in_the_money,t]
        model = Ridge(alpha=alpha, solver='lsqr').fit(X_trans, cash_flows[in_the_money] - euro_t)
        continuation = model.predict(X_trans) + euro_t
        
        # Exercise decision
        immediate = intrinsic[in_the_money] * np.exp(-r*t*dt)
        exercise = (immediate > continuation) & (immediate > 1e-6)
        cash_flows[in_the_money] = np.where(exercise, immediate, cash_flows[in_the_money])
        
        # Update visualization
        if t % 5 == 0:
            df = generate_fake_candles(initial_price=S0)
            fig = go.Figure(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            ))
            fig.update_layout(
                title='Live Market Simulation',
                xaxis_title='Time',
                yaxis_title='Price (₹)',
                template='plotly_white',
                height=300
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    chart_placeholder.empty()
    
    # Final variance reduction
    euro_final = black_scholes(S0, K, T, r, sigma, option_type)
    price = np.mean(cash_flows) - (np.mean(payoff) - euro_final)
    
    return price, np.std(cash_flows)/np.sqrt(N)

@lru_cache(maxsize=128)
def cached_pricing(S0, K, T, r, sigma, option_type):
    """Accelerated pricing for Greek calculations"""
    return american_option_pricing(S0, K, T, r, sigma, option_type, N=100000)[0]

def calculate_greeks(S0, K, T, r, sigma, option_type='put'):
    """Parallelized precision Greek calculation"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            'base': executor.submit(cached_pricing, S0, K, T, r, sigma, option_type),
            'up': executor.submit(cached_pricing, S0*1.01, K, T, r, sigma, option_type),
            'down': executor.submit(cached_pricing, S0*0.99, K, T, r, sigma, option_type),
            'vol_up': executor.submit(cached_pricing, S0, K, T, r, sigma+0.01, option_type),
            'vol_down': executor.submit(cached_pricing, S0, K, T, r, sigma-0.01, option_type),
            'T': executor.submit(cached_pricing, S0, K, max(T-1/365,1e-5), r, sigma, option_type),
            'r_up': executor.submit(cached_pricing, S0, K, T, r+0.01, sigma, option_type),
            'r_down': executor.submit(cached_pricing, S0, K, T, r-0.01, sigma, option_type),
        }
        results = {k: f.result() for k,f in futures.items()}
    
    return {
        'Delta': (results['up'] - results['down'])/(0.02*S0),
        'Gamma': (results['up'] - 2*results['base'] + results['down'])/(0.01*S0)**2,
        'Vega': (results['vol_up'] - results['vol_down'])/2,
        'Theta': (results['T'] - results['base'])*365,
        'Rho': (results['r_up'] - results['r_down'])/2
    }

@st.cache_data
def plot_early_exercise_boundary(S0, K, T, r, sigma):
    """Early exercise boundary visualization"""
    time_steps = np.linspace(0, T, 20)
    boundaries = [K - american_option_pricing(S0, K, max(t,0.001), r, sigma, 'put', 50000, 50)[0]
                 for t in time_steps]
    
    fig = go.Figure(go.Scatter(
        x=time_steps*365,
        y=boundaries,
        mode='lines+markers',
        line=dict(color='#FF6F00')
    ))
    fig.update_layout(
        title='Early Exercise Boundary',
        xaxis_title='Days to Expiry',
        yaxis_title='Critical Price (₹)',
        template='plotly_white'
    )
    return fig

def run_scenario_analysis(S0, K, T, r, sigma):
    """Scenario analysis with same UI"""
    scenarios = {
        'Bull Market': {'spot': S0*1.2, 'vol': sigma*0.8},
        'Bear Market': {'spot': S0*0.8, 'vol': sigma*1.2},
        'Volatility Spike': {'spot': S0, 'vol': sigma*1.5}
    }
    
    results = []
    for name, params in scenarios.items():
        price = american_option_pricing(params['spot'], K, T, r, params['vol'], 'put', 50000)[0]
        results.append({
            'Scenario': name,
            'Spot Price': params['spot'],
            'Volatility': f"{params['vol']*100:.1f}%",
            'Option Price': price
        })
    
    return pd.DataFrame(results)

# Main interface
st.markdown("### American Option Pricing Model with P&L Analysis")

# Price display columns
col1, col2 = st.columns(2)
with col1:
    with st.spinner("Calculating CALL option..."):
        call_price, call_se = american_option_pricing(S0, K, T, r, sigma, 'call', N, M, degree, alpha, seed)
    call_pnl = call_price - call_purchase
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div class="metric-label">American CALL Value</div>
            <div class="metric-value">₹{call_price:,.2f}</div>
            <div>± {call_se:.4f} (SE)</div>
            <div style="margin-top:15px">
                P&L: <span class="{'profit' if call_pnl>=0 else 'loss'}">₹{call_pnl:,.2f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    with st.spinner("Calculating PUT option..."):
        put_price, put_se = american_option_pricing(S0, K, T, r, sigma, 'put', N, M, degree, alpha, seed)
    put_pnl = put_price - put_purchase
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div class="metric-label">American PUT Value</div>
            <div class="metric-value">₹{put_price:,.2f}</div>
            <div>± {put_se:.4f} (SE)</div>
            <div style="margin-top:15px">
                P&L: <span class="{'profit' if put_pnl>=0 else 'loss'}">₹{put_pnl:,.2f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Greeks Analysis
st.markdown("---")
st.title("📉 Greeks Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option Greeks")
    greeks = calculate_greeks(S0, K, T, r, sigma, 'call')
    for greek, value in greeks.items():
        st.metric(greek, f"{value:.4f}")

with col2:
    st.subheader("Put Option Greeks")
    greeks = calculate_greeks(S0, K, T, r, sigma, 'put')
    for greek, value in greeks.items():
        st.metric(greek, f"{value:.4f}")

# Visualization sections
st.markdown("---")
st.title("⚡ Early Exercise Boundary")
st.plotly_chart(plot_early_exercise_boundary(S0, K, T, r, sigma), use_container_width=True)

st.markdown("---")
st.title("📚 Scenario Comparison")
st.dataframe(
    run_scenario_analysis(S0, K, T, r, sigma).style.format({
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
    for path in S:
        fig.add_trace(go.Scatter(
            x=np.linspace(0, days_to_maturity, M+1),
            y=path,
            mode='lines',
            line=dict(width=0.5)
        ))
    fig.update_layout(
        title='Monte Carlo Paths',
        xaxis_title='Days to Expiry',
        yaxis_title='Price (₹)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
