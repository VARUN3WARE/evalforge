import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def save_plot(fig, filename, output_dir="reports"):
    """
    Saves a matplotlib figure to the output directory.
    Because if a plot isn't saved as a PNG, did it really happen? :)
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return filepath

def plot_fragility_drop(baseline_acc, drops_dict, output_dir="reports"):
    """
    Plots the baseline accuracy vs the performance drops caused by our adversarial attacks.
    """
    if not drops_dict:
        return None
        
    labels = ["Baseline"] + list(drops_dict.keys())
    
    # Drops are absolute drops. So perturbed_score = baseline - drop
    scores = [baseline_acc * 100] + [(baseline_acc - drop) * 100 for drop in drops_dict.values()]
    colors = ['#2ecc71'] + ['#e74c3c'] * len(drops_dict)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, scores, color=colors)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Adversarial Fragility: Performance Under Stress')
    ax.set_ylim([max(0, min(scores) - 10), 105])
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.xticks(rotation=15)
    return save_plot(fig, "fragility_drop.png", output_dir)

def plot_drift_histogram(train_df, test_df, feature_name, output_dir="reports"):
    """
    Plots the overlapping distributions of a specific feature to visually prove drift.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(train_df[feature_name].dropna(), bins=30, alpha=0.5, label='Train', color='#3498db', density=True)
    ax.hist(test_df[feature_name].dropna(), bins=30, alpha=0.5, label='Test', color='#e74c3c', density=True)
    
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Density')
    ax.set_title(f'Data Drift Detected: {feature_name}')
    ax.legend(loc='upper right')
    
    return save_plot(fig, f"drift_{feature_name}.png", output_dir)
