import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="TabPFN Research Implementation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #616161;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤖 TabPFN Research Implementation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Transformer That Solves Small Tabular Classification Problems in a Second</div>', unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'experiment_run' not in st.session_state:
    st.session_state.experiment_run = False

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Device Selection
st.sidebar.subheader("🖥️ Device Selection")
cuda_available = torch.cuda.is_available()

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    st.sidebar.success(f"✅ GPU Available: {gpu_name}")
    st.sidebar.info(f"💾 GPU Memory: {gpu_memory:.2f} GB")
    
    device_option = st.sidebar.radio(
        "Select Compute Device:",
        ["GPU (CUDA)", "CPU"],
        index=0,
        help="GPU provides significant speedup for TabPFN"
    )
    device = 'cuda' if device_option == "GPU (CUDA)" else 'cpu'
else:
    st.sidebar.warning("⚠️ No GPU detected - using CPU")
    device = 'cpu'
    device_option = "CPU"

st.sidebar.markdown(f"**Current Device:** `{device.upper()}`")

# Dataset Selection
st.sidebar.subheader("📊 Dataset Selection")
dataset_options = st.sidebar.multiselect(
    "Choose Datasets:",
    ["Iris", "Wine", "Breast Cancer", "Digits"],
    default=["Iris", "Wine", "Breast Cancer"],
    help="Select datasets for comparison (all meet TabPFN constraints)"
)

# Model Selection
st.sidebar.subheader("🎯 Model Selection")
model_options = st.sidebar.multiselect(
    "Choose Models to Compare:",
    ["TabPFN (GPU)", "TabPFN (CPU)", "XGBoost", "LightGBM", "Random Forest", "Logistic Regression"],
    default=["TabPFN (GPU)", "XGBoost", "LightGBM"] if cuda_available else ["TabPFN (CPU)", "XGBoost"],
    help="Select models for benchmarking"
)

# Experiment Parameters
st.sidebar.subheader("🔬 Experiment Parameters")
test_size = st.sidebar.slider(
    "Test Set Size (%)",
    min_value=20,
    max_value=50,
    value=30,
    step=5,
    help="Percentage of data to use for testing"
) / 100

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=1000,
    value=42,
    help="For reproducible results"
)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Information tabs
tab1, tab2, tab3, tab4 = st.tabs(["📖 About", "🚀 Run Experiment", "📊 Results", "📈 Visualizations"])

# ============================================================================
# TAB 1: About
# ============================================================================

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎓 Research Paper")
        st.markdown("""
        **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second**
        
        - **Authors:** Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter
        - **Published:** ICLR 2023
        - **Paper:** [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
        - **Code:** [GitHub](https://github.com/automl/TabPFN)
        """)
        
        st.markdown("### 🎯 Key Innovations")
        st.markdown("""
        1. **In-Context Learning:** No training needed at test time
        2. **Prior-Data Fitted Networks:** Pre-trained on synthetic datasets
        3. **Bayesian Inference:** Approximates posterior predictive distribution
        4. **Zero Hyperparameter Tuning:** Works out-of-the-box
        5. **GPU Acceleration:** 5,700× speedup reported in paper
        """)
    
    with col2:
        st.markdown("### ⚡ TabPFN Constraints")
        st.info("""
        **For optimal performance:**
        - ✅ Training samples: ≤ 1,000
        - ✅ Features: ≤ 100 (numerical only)
        - ✅ Classes: ≤ 10
        - ✅ No missing values
        - ⚠️ Best for numerical data
        """)
        
        st.markdown("### 📊 Datasets Used")
        st.markdown("""
        All datasets meet TabPFN constraints:
        
        - **Iris:** 150 samples, 4 features, 3 classes
        - **Wine:** 178 samples, 13 features, 3 classes
        - **Breast Cancer:** 569 samples, 30 features, 2 classes
        - **Digits:** 500 samples, 64 features, 10 classes
        """)

# ============================================================================
# TAB 2: Run Experiment
# ============================================================================

with tab2:
    st.markdown("### 🚀 Run Comparative Analysis")
    
    # Show selected configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Device", device.upper())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Datasets", len(dataset_options))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models", len(model_options))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Run button
    if st.button("▶️ Run Experiment", type="primary", use_container_width=True):
        if not dataset_options:
            st.error("❌ Please select at least one dataset!")
        elif not model_options:
            st.error("❌ Please select at least one model!")
        else:
            # Import required libraries
            try:
                from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from tabpfn import TabPFNClassifier
                
                # Optional imports
                try:
                    import xgboost as xgb
                    xgb_available = True
                except:
                    xgb_available = False
                    if "XGBoost" in model_options:
                        st.warning("⚠️ XGBoost not installed")
                
                try:
                    import lightgbm as lgb
                    lgb_available = True
                except:
                    lgb_available = False
                    if "LightGBM" in model_options:
                        st.warning("⚠️ LightGBM not installed")
                
            except ImportError as e:
                st.error(f"❌ Required library not installed: {e}")
                st.stop()
            
            # Load datasets
            datasets = {}
            if "Iris" in dataset_options:
                iris = load_iris()
                datasets['Iris'] = (iris.data, iris.target)
            
            if "Wine" in dataset_options:
                wine = load_wine()
                datasets['Wine'] = (wine.data, wine.target)
            
            if "Breast Cancer" in dataset_options:
                cancer = load_breast_cancer()
                datasets['Breast Cancer'] = (cancer.data, cancer.target)
            
            if "Digits" in dataset_options:
                digits = load_digits()
                datasets['Digits'] = (digits.data[:500], digits.target[:500])
            
            # Run experiments
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_experiments = len(datasets) * len(model_options)
            current_experiment = 0
            
            for dataset_name, (X, y) in datasets.items():
                status_text.text(f"Processing dataset: {dataset_name}...")
                
                # Preprocess
                X = pd.DataFrame(X).fillna(0).values
                y = LabelEncoder().fit_transform(y)
                X = StandardScaler().fit_transform(X)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=random_seed
                )
                
                # Helper function for ROC AUC
                def safe_roc_auc(y_true, y_prob):
                    try:
                        if len(np.unique(y_true)) == 2:
                            return roc_auc_score(y_true, y_prob[:, 1])
                        else:
                            return roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
                    except:
                        return np.nan
                
                # Test each model
                for model_name in model_options:
                    status_text.text(f"Testing {model_name} on {dataset_name}...")
                    
                    try:
                        start_time = time.time()
                        
                        if model_name == "TabPFN (GPU)":
                            model = TabPFNClassifier(device='cuda' if cuda_available else 'cpu')
                        elif model_name == "TabPFN (CPU)":
                            model = TabPFNClassifier(device='cpu')
                        elif model_name == "XGBoost" and xgb_available:
                            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=random_seed)
                        elif model_name == "LightGBM" and lgb_available:
                            model = lgb.LGBMClassifier(verbose=-1, random_state=random_seed)
                        elif model_name == "Random Forest":
                            model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
                        elif model_name == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000, random_state=random_seed)
                        else:
                            current_experiment += 1
                            continue
                        
                        # Train
                        model.fit(X_train, y_train)
                        
                        # Predict
                        y_pred_proba = model.predict_proba(X_test)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        
                        elapsed_time = time.time() - start_time
                        
                        # Metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        roc_auc = safe_roc_auc(y_test, y_pred_proba)
                        
                        results.append({
                            'Dataset': dataset_name,
                            'Model': model_name,
                            'Accuracy': accuracy,
                            'ROC_AUC': roc_auc,
                            'Time (s)': elapsed_time,
                            'Samples': len(X_train),
                            'Features': X_train.shape[1],
                            'Classes': len(np.unique(y_train))
                        })
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error with {model_name} on {dataset_name}: {str(e)}")
                    
                    current_experiment += 1
                    progress_bar.progress(current_experiment / total_experiments)
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state.results_df = pd.DataFrame(results)
            st.session_state.experiment_run = True
            
            st.success("✅ Experiment completed successfully!")
            st.balloons()

# ============================================================================
# TAB 3: Results
# ============================================================================

with tab3:
    st.markdown("### 📊 Experimental Results")
    
    if st.session_state.experiment_run and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        
        # Summary statistics
        st.markdown("#### 📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Experiments", len(results_df))
        
        with col2:
            best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
            st.metric("Best Model", best_model)
        
        with col3:
            fastest_model = results_df.loc[results_df['Time (s)'].idxmin(), 'Model']
            st.metric("Fastest Model", fastest_model)
        
        with col4:
            avg_acc = results_df['Accuracy'].mean()
            st.metric("Avg Accuracy", f"{avg_acc:.4f}")
        
        st.markdown("---")
        
        # Detailed results table
        st.markdown("#### 📋 Detailed Results")
        
        # Format the dataframe
        display_df = results_df.copy()
        display_df['Accuracy'] = display_df['Accuracy'].map('{:.4f}'.format)
        display_df['ROC_AUC'] = display_df['ROC_AUC'].map(lambda x: '{:.4f}'.format(x) if not pd.isna(x) else 'N/A')
        display_df['Time (s)'] = display_df['Time (s)'].map('{:.4f}'.format)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Average performance by model
        st.markdown("#### 🏆 Average Performance by Model")
        
        avg_by_model = results_df.groupby('Model').agg({
            'Accuracy': 'mean',
            'ROC_AUC': 'mean',
            'Time (s)': 'mean'
        }).round(4).sort_values('Accuracy', ascending=False)
        
        st.dataframe(avg_by_model, use_container_width=True)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="tabpfn_experiment_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("👈 Run an experiment first to see results!")

# ============================================================================
# TAB 4: Visualizations
# ============================================================================

with tab4:
    st.markdown("### 📈 Performance Visualizations")
    
    if st.session_state.experiment_run and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        
        # Plot 1: Accuracy comparison
        st.markdown("#### 🎯 Accuracy Comparison")
        
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Grouped bar chart
        datasets_list = results_df['Dataset'].unique()
        models_list = results_df['Model'].unique()
        x = np.arange(len(datasets_list))
        width = 0.8 / len(models_list)
        
        for i, model in enumerate(models_list):
            model_data = results_df[results_df['Model'] == model]
            accuracies = [model_data[model_data['Dataset'] == ds]['Accuracy'].values[0] 
                         if len(model_data[model_data['Dataset'] == ds]) > 0 else 0 
                         for ds in datasets_list]
            ax1.bar(x + i * width, accuracies, width, label=model, alpha=0.8)
        
        ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy Comparison Across Datasets', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(models_list) - 1) / 2)
        ax1.set_xticklabels(datasets_list, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0.8, 1.0])
        
        st.pyplot(fig1)
        
        # Plot 2: Time comparison
        st.markdown("#### ⏱️ Runtime Comparison")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        avg_time = results_df.groupby('Model')['Time (s)'].mean().sort_values()
        colors = ['red' if 'TabPFN' in x else 'steelblue' for x in avg_time.index]
        
        ax2.barh(range(len(avg_time)), avg_time.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(avg_time)))
        ax2.set_yticklabels(avg_time.index)
        ax2.set_xlabel('Average Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Runtime by Model', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig2)
        
        # Plot 3: Accuracy vs Time scatter
        st.markdown("#### 🎯 Accuracy vs Runtime Trade-off")
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            color = 'red' if 'TabPFN' in model else 'steelblue'
            size = 200 if 'TabPFN' in model else 100
            ax3.scatter(model_data['Time (s)'], model_data['Accuracy'], 
                       s=size, alpha=0.6, c=color, label=model)
        
        ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Accuracy vs Runtime (Fast & Accurate = Top-Left)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_xscale('log')
        
        st.pyplot(fig3)
        
        # Plot 4: GPU vs CPU speedup (if applicable)
        if 'TabPFN (GPU)' in results_df['Model'].values and 'TabPFN (CPU)' in results_df['Model'].values:
            st.markdown("#### 🚀 GPU vs CPU Speedup")
            
            speedups = []
            dataset_names = []
            
            for dataset in results_df['Dataset'].unique():
                gpu_time = results_df[(results_df['Dataset'] == dataset) & 
                                     (results_df['Model'] == 'TabPFN (GPU)')]['Time (s)'].values
                cpu_time = results_df[(results_df['Dataset'] == dataset) & 
                                     (results_df['Model'] == 'TabPFN (CPU)')]['Time (s)'].values
                
                if len(gpu_time) > 0 and len(cpu_time) > 0:
                    speedup = cpu_time[0] / gpu_time[0]
                    speedups.append(speedup)
                    dataset_names.append(dataset)
            
            if speedups:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                
                ax4.bar(range(len(speedups)), speedups, color='green', alpha=0.7)
                ax4.set_xticks(range(len(speedups)))
                ax4.set_xticklabels(dataset_names, rotation=45, ha='right')
                ax4.set_ylabel('Speedup Factor (×)', fontsize=12, fontweight='bold')
                ax4.set_title('GPU vs CPU Speedup for TabPFN', fontsize=14, fontweight='bold')
                ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
                ax4.grid(axis='y', alpha=0.3)
                ax4.legend()
                
                avg_speedup = np.mean(speedups)
                st.pyplot(fig4)
                st.success(f"✅ Average GPU Speedup: **{avg_speedup:.2f}×**")
        
    else:
        st.info("👈 Run an experiment first to see visualizations!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>TabPFN Research Implementation</strong></p>
    <p>Paper: <a href='https://arxiv.org/abs/2207.01848'>arXiv:2207.01848</a> | 
    Code: <a href='https://github.com/automl/TabPFN'>GitHub</a></p>
    <p>Implementation for ML Lab Mini Project</p>
</div>
""", unsafe_allow_html=True)