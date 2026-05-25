import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Import the configuration from your project
from config import TrainConfig

import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

def smooth_curve(values, smoothing=0.9):
    """
    TensorBoard-style exponential moving average smoothing.

    smoothing:
        0.0 = no smoothing
        0.99 = very smooth
    """
    if len(values) == 0:
        return values

    smoothed = []
    last = values[0]

    for v in values:
        last = last * smoothing + (1 - smoothing) * v
        smoothed.append(last)

    return np.array(smoothed)

def plot_metric_chart(df, title, color, smoothing=0.9):
    """
    Plot raw + smoothed curves using Plotly.
    """
    steps = df.index.to_numpy()
    values = df.iloc[:, 0].to_numpy()

    smooth_values = smooth_curve(values, smoothing)

    fig = go.Figure()

    # Raw curve
    fig.add_trace(go.Scatter(
        x=steps,
        y=values,
        mode="lines",
        name="Raw",
        opacity=0.25,
    ))

    # Smoothed curve
    fig.add_trace(go.Scatter(
        x=steps,
        y=smooth_values,
        mode="lines",
        name="Smoothed",
        line=dict(width=3, color=color),
    ))

    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    fig.update_xaxes(title="Step")
    fig.update_yaxes(title="Value")

    st.plotly_chart(fig, use_container_width=True)

# Page configuration (Must be the first Streamlit call)
st.set_page_config(
    page_title="LoRA Training Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Data Loading & Processing (with caching)
# ==========================================

@st.cache_data(ttl=5)
def get_config_dict() -> dict:
    """Dynamically read and return the latest configuration dictionary."""
    cfg = TrainConfig()
    return vars(cfg)

@st.cache_data(ttl=5)
def load_tensorboard_data(log_dir: str) -> dict:
    """Read TensorBoard logs and return metrics as a dictionary of DataFrames."""
    if not os.path.exists(log_dir):
        return {}
    
    # Find the latest event file
    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        return {}
    
    # Get the directory of the most recently modified log file
    latest_log_dir = str(max(event_files, key=os.path.getmtime).parent)
    
    # Set size_guidance for scalars to 0 to load all records
    ea = EventAccumulator(latest_log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    
    metrics = {}
    if 'scalars' in ea.Tags():
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            df = pd.DataFrame([(e.step, e.value) for e in events], columns=["Step", tag])
            df = df.set_index("Step")
            metrics[tag] = df
            
    return metrics

@st.cache_data(ttl=5)
def get_sample_images(output_dir: str, output_name: str) -> list:
    """Retrieve all generated sample images, sorted by modification time (newest first)."""
    sample_dir = Path(output_dir) / f"{output_name}_samples"
    if not sample_dir.exists():
        return []
    
    images = list(sample_dir.glob("*.png"))
    images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return images

# ==========================================
# UI Construction
# ==========================================

def main():
    cfg_dict = get_config_dict()
    
    # --- Sidebar: Training Parameters & Controls ---
    with st.sidebar:
        st.title("⚙️ Control Panel")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("Enable Auto-Refresh (10s)", value=True)

        smoothing = st.slider(
            "Curve Smoothing",
            min_value=0.0,
            max_value=0.99,
            value=0.90,
            step=0.01,
            help="TensorBoard-style EMA smoothing"
        )
        
        st.divider()
        st.subheader("Training Parameters")
        
        # Extract and display key parameters
        st.metric("Dataset", os.path.basename(cfg_dict.get('train_data_dir', 'N/A')))
        st.metric("Target Output", cfg_dict.get('output_name', 'N/A'))
        st.metric("Base Model", os.path.basename(cfg_dict.get('pretrained_model_name_or_path', 'N/A')))
        
        # Expandable full configuration tree
        with st.expander("View Full Config", expanded=False):
            st.json(cfg_dict)

    # --- Main Area ---
    st.title("📈 LoRA Training Dashboard")
    
    # Append output_name to the logging directory as requested
    target_log_dir = os.path.join(cfg_dict['logging_dir'], cfg_dict['output_name'])
    st.caption(f"Logging Directory: `{target_log_dir}` | Output Directory: `{cfg_dict['output_dir']}`")

    # 1. Top Metrics Overview
    metrics_data = load_tensorboard_data(target_log_dir)
    
    if metrics_data:
        st.subheader("Real-time Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        if "Train/Loss" in metrics_data and not metrics_data["Train/Loss"].empty:
            latest_step = metrics_data["Train/Loss"].index[-1]
            latest_loss = metrics_data["Train/Loss"].iloc[-1].values[0]
            col1.metric("Current Step", f"{latest_step:,}")
            col2.metric("Latest Loss", f"{latest_loss:.4f}")
            
        if "LR/Effective_Actual_LR" in metrics_data and not metrics_data["LR/Effective_Actual_LR"].empty:
            latest_lr = metrics_data["LR/Effective_Actual_LR"].iloc[-1].values[0]
            col3.metric("Effective LR", f"{latest_lr:.2e}")

        if "LR/Prodigy_D_Factor" in metrics_data and not metrics_data["LR/Prodigy_D_Factor"].empty:
            latest_d = metrics_data["LR/Prodigy_D_Factor"].iloc[-1].values[0]
            col4.metric("Prodigy D Factor", f"{latest_d:.4f}")

        st.divider()

        # 2. Plotting Charts (2x2 Grid)
        st.subheader("Training Charts")
        
        # Row 1
        r1_col1, r1_col2 = st.columns(2)
        with r1_col1:
            if "Train/Loss" in metrics_data:
                st.markdown("**Train / Loss**")
                plot_metric_chart(metrics_data["Train/Loss"], "Train / Loss", "#ff4b4b", smoothing)
                
        with r1_col2:
            if "LR/Effective_Actual_LR" in metrics_data:
                st.markdown("**LR / Effective Actual LR**")
                plot_metric_chart(metrics_data["LR/Effective_Actual_LR"],"LR / Effective Actual LR", "#0068c9",smoothing)

        # Row 2
        r2_col1, r2_col2 = st.columns(2)
        with r2_col1:
            if "LR/Base_Scheduled" in metrics_data:
                st.markdown("**LR / Base Scheduled**")
                plot_metric_chart(metrics_data["LR/Base_Scheduled"],"LR / Base Scheduled","#00c968", smoothing)
                
        with r2_col2:
            if "LR/Prodigy_D_Factor" in metrics_data:
                st.markdown("**LR / Prodigy D Factor**")
                plot_metric_chart(metrics_data["LR/Prodigy_D_Factor"],"LR / Prodigy D Factor", "#c900c9",smoothing)
    else:
        st.info("No TensorBoard logs found yet. Waiting for training to start logging...")

    st.divider()

    # 3. Generated Samples Scrollable Grid
    st.subheader("Generated Samples")
    images = get_sample_images(cfg_dict['output_dir'], cfg_dict['output_name'])
    
    if images:
        # Create a fixed-height scrollable container
        with st.container(height=650):
            cols = st.columns(3)
            for i, img_path in enumerate(images):
                with cols[i % 3]:
                    # Use use_container_width=True to fix the deprecation warning
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.info("No sample images generated yet.")

    # --- Auto-refresh Logic ---
    if auto_refresh:
        st_autorefresh(interval=10_000, key="training_refresh")

if __name__ == "__main__":
    main()