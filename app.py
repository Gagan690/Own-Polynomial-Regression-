import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Polynomial Regression Learning",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .metric-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Interactive Polynomial Regression Learning</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore how polynomial regression works by adjusting parameters and observing the results</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Dataset Parameters")

# Input parameters
n_points = st.sidebar.slider("Number of Data Points", min_value=10, max_value=500, value=100, step=10)
noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
slope = st.sidebar.slider("True Slope", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
intercept = st.sidebar.slider("True Intercept", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)

st.sidebar.header("Regression Parameters")
poly_degree = st.sidebar.slider("Polynomial Degree", min_value=1, max_value=10, value=2, step=1)

# Generate data button
generate_data = st.sidebar.button("Generate New Dataset", type="primary")

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

# Generate data
if generate_data or not st.session_state.data_generated:
    try:
        # Generate random x values
        np.random.seed(42)  # For reproducibility, remove in production
        x = np.linspace(-5, 5, n_points)
        
        # Generate y values with noise
        y_true = intercept + slope * x
        noise = np.random.normal(0, noise_level, n_points)
        y = y_true + noise
        
        # Store in session state
        st.session_state.x = x
        st.session_state.y = y
        st.session_state.y_true = y_true
        st.session_state.data_generated = True
        
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")

# Perform polynomial regression if data exists
if st.session_state.data_generated:
    try:
        x = st.session_state.x
        y = st.session_state.y
        y_true = st.session_state.y_true
        
        # Check if polynomial degree is valid
        if poly_degree >= len(x):
            st.error(f"Polynomial degree ({poly_degree}) must be less than the number of data points ({len(x)})")
        else:
            # Fit polynomial regression
            poly_features = PolynomialFeatures(degree=poly_degree)
            x_poly = poly_features.fit_transform(x.reshape(-1, 1))
            
            model = LinearRegression()
            model.fit(x_poly, y)
            
            # Generate predictions
            x_plot = np.linspace(x.min(), x.max(), 300)
            x_plot_poly = poly_features.transform(x_plot.reshape(-1, 1))
            y_pred_plot = model.predict(x_plot_poly)
            
            # Calculate predictions for original points
            y_pred = model.predict(x_poly)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Find optimal polynomial degree (minimum MSE)
            max_degree = min(10, len(x) - 1)  # Ensure we don't exceed data points
            degrees_to_test = range(1, max_degree + 1)
            mse_scores = []
            
            for degree in degrees_to_test:
                poly_features_test = PolynomialFeatures(degree=degree)
                x_poly_test = poly_features_test.fit_transform(x.reshape(-1, 1))
                model_test = LinearRegression()
                model_test.fit(x_poly_test, y)
                y_pred_test = model_test.predict(x_poly_test)
                mse_test = mean_squared_error(y, y_pred_test)
                mse_scores.append(mse_test)
            
            # Find optimal degree
            optimal_degree = degrees_to_test[np.argmin(mse_scores)]
            optimal_mse = min(mse_scores)
            
            # Calculate optimal polynomial for plotting
            poly_features_optimal = PolynomialFeatures(degree=optimal_degree)
            x_poly_optimal = poly_features_optimal.fit_transform(x.reshape(-1, 1))
            model_optimal = LinearRegression()
            model_optimal.fit(x_poly_optimal, y)
            
            # Generate predictions for optimal model
            x_plot_optimal = poly_features_optimal.transform(x_plot.reshape(-1, 1))
            y_pred_plot_optimal = model_optimal.predict(x_plot_optimal)
            
            # Create two columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot original data points
                ax.scatter(x, y, alpha=0.6, color='#3b82f6', s=50, label='Data Points')
                
                # Plot true relationship (if linear)
                if poly_degree == 1:
                    ax.plot(x, y_true, color='#10b981', linewidth=2, linestyle='--', label='True Relationship')
                
                # Plot polynomial fit
                ax.plot(x_plot, y_pred_plot, color='#ef4444', linewidth=3, label=f'Polynomial Fit (degree {poly_degree})')
                
                # Plot optimal polynomial fit
                ax.plot(x_plot, y_pred_plot_optimal, color='#f59e0b', linewidth=3, linestyle='-.', 
                       label=f'Optimal Fit (degree {optimal_degree}, MSE: {optimal_mse:.4f})')
                
                ax.set_xlabel('X', fontsize=12)
                ax.set_ylabel('Y', fontsize=12)
                ax.set_title(f'Polynomial Regression (Degree {poly_degree})', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Style the plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)
            
            with col2:
                st.subheader("Model Performance")
                
                # Display metrics in styled cards
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Mean Squared Error (MSE)</div>
                    <div class="metric-value">{mse:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">R² Score</div>
                    <div class="metric-value">{r2:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Data Points</div>
                    <div class="metric-value">{n_points}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Polynomial Degree</div>
                    <div class="metric-value">{poly_degree}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Optimal Degree (Min MSE)</div>
                    <div class="metric-value">{optimal_degree}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Optimal MSE</div>
                    <div class="metric-value">{optimal_mse:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Model coefficients
                st.subheader("Model Coefficients")
                coefficients = np.polyfit(x, y, poly_degree)
                for i, coef in enumerate(coefficients):
                    power = poly_degree - i
                    if power == 0:
                        st.write(f"Constant: {coef:.4f}")
                    elif power == 1:
                        st.write(f"x¹: {coef:.4f}")
                    else:
                        st.write(f"x^{power}: {coef:.4f}")
            
            # Educational content
            st.markdown("---")
            st.subheader("Understanding the Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Mean Squared Error (MSE):**
                - Measures the average squared difference between actual and predicted values
                - Lower values indicate better fit
                - MSE = 0 means perfect fit (rarely achievable with real data)
                """)
                
                st.markdown("""
                **R² Score (Coefficient of Determination):**
                - Indicates the proportion of variance explained by the model
                - Range: 0 to 1 (higher is better)
                - R² = 1 means perfect fit
                - R² = 0 means the model is no better than predicting the mean
                """)
            
            with col2:
                st.markdown("""
                **Polynomial Degree:**
                - Degree 1: Linear regression (straight line)
                - Degree 2: Quadratic (parabola)
                - Higher degrees: More complex curves
                - **Warning:** Very high degrees can lead to overfitting
                """)
                
                st.markdown("""
                **Tips for Learning:**
                - Try different noise levels to see how it affects the fit
                - Experiment with polynomial degrees
                - Notice how higher degrees can overfit to noise
                - Compare R² and MSE values across different settings
                """)
    
    except Exception as e:
        st.error(f"Error in polynomial regression: {str(e)}")

else:
    st.info("Click 'Generate New Dataset' to start exploring polynomial regression!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
    Built with Streamlit • Interactive Learning Tool for Polynomial Regression
</div>
""", unsafe_allow_html=True)
