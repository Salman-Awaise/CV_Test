import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import load_images
from baselines.classical import (
    gray_world_estimate, white_patch_estimate, shades_of_gray_estimate,
    max_rgb_estimate, edge_based_estimate, robust_awb_estimate
)
from evaluation.metrics import angular_error, compute_statistics

def evaluate_algorithms_batch(image_dir: str, ground_truth_dir: str = None) -> Dict[str, List[float]]:
    """
    Evaluate all color constancy algorithms on a batch of images.
    """
    # Load images
    images = load_images(image_dir)
    
    # If no ground truth directory provided, use gray world on color-corrected versions
    if ground_truth_dir is None:
        # Assume we have building1.npy and building1_cc.npy pattern
        ground_truths = []
        for i, img in enumerate(images):
            # For demo, use a synthetic ground truth
            gt = np.array([0.95, 1.0, 1.05])
            gt = gt / np.linalg.norm(gt)
            ground_truths.append(gt)
    else:
        ground_truths = load_images(ground_truth_dir)
        ground_truths = [gray_world_estimate(gt) for gt in ground_truths]
    
    # Define algorithms to test
    algorithms = {
        'Gray World': gray_world_estimate,
        'White Patch': white_patch_estimate,
        'Shades of Gray (p=6)': lambda img: shades_of_gray_estimate(img, p=6),
        'Shades of Gray (p=1)': lambda img: shades_of_gray_estimate(img, p=1),
        'Max RGB': max_rgb_estimate,
        'Edge-based': edge_based_estimate,
        'Robust AWB (95%)': lambda img: robust_awb_estimate(img, percentile=95),
        'Robust AWB (90%)': lambda img: robust_awb_estimate(img, percentile=90),
    }
    
    # Evaluate each algorithm
    results = {}
    for algo_name, algo_func in algorithms.items():
        errors = []
        for img, gt in zip(images, ground_truths):
            try:
                pred = algo_func(img)
                error = angular_error(pred, gt)
                errors.append(error)
            except Exception as e:
                print(f"Error in {algo_name}: {e}")
                errors.append(np.inf)
        results[algo_name] = errors
    
    return results

def generate_comprehensive_statistics(results: Dict[str, List[float]]) -> None:
    """
    Generate comprehensive statistics and visualizations for all algorithms.
    """
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Compute statistics for each algorithm
    stats_summary = {}
    for algo_name, errors in results.items():
        valid_errors = [e for e in errors if not np.isinf(e)]
        if valid_errors:
            stats = compute_statistics(valid_errors)
            stats_summary[algo_name] = stats
    
    # Save detailed statistics
    with open('results/comprehensive_stats.txt', 'w') as f:
        f.write("Comprehensive Color Constancy Algorithm Comparison\\n")
        f.write("=" * 60 + "\\n\\n")
        
        for algo_name, stats in stats_summary.items():
            f.write(f"{algo_name}:\\n")
            f.write("-" * 30 + "\\n")
            for metric, value in stats.items():
                f.write(f"  {metric}: {value:.3f}\\n")
            f.write("\\n")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of mean errors
    algo_names = list(stats_summary.keys())
    mean_errors = [stats_summary[name]['mean'] for name in algo_names]
    
    bars = ax1.bar(range(len(algo_names)), mean_errors, color='skyblue')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Mean Angular Error (degrees)')
    ax1.set_title('Mean Angular Error Comparison')
    ax1.set_xticks(range(len(algo_names)))
    ax1.set_xticklabels(algo_names, rotation=45, ha='right')
    
    # Add target line at 0.9 degrees
    ax1.axhline(y=0.9, color='red', linestyle='--', label='Target (0.9°)')
    ax1.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_errors):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Box plot of error distributions
    error_lists = [results[name] for name in algo_names]
    ax2.boxplot(error_lists, labels=algo_names)
    ax2.set_ylabel('Angular Error (degrees)')
    ax2.set_title('Error Distribution Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0.9, color='red', linestyle='--', label='Target (0.9°)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed performance table
    performance_table = []
    for algo_name in algo_names:
        stats = stats_summary[algo_name]
        performance_table.append([
            algo_name,
            f"{stats['mean']:.3f}",
            f"{stats['median']:.3f}",
            f"{stats['perc95']:.3f}",
            f"{stats['pct_lt3']:.1f}%",
            f"{stats['pct_lt5']:.1f}%"
        ])
    
    # Save performance table
    with open('results/performance_table.csv', 'w') as f:
        f.write("Algorithm,Mean Error,Median Error,95th Percentile,<3° (%),<5° (%)\\n")
        for row in performance_table:
            f.write(",".join(row) + "\\n")
    
    print("Comprehensive evaluation completed!")
    print(f"Results saved to results/ directory")
    print(f"Best performing algorithm: {min(stats_summary.keys(), key=lambda x: stats_summary[x]['mean'])}")

if __name__ == "__main__":
    # Evaluate on available images
    print("Running comprehensive algorithm evaluation...")
    results = evaluate_algorithms_batch('data/')
    generate_comprehensive_statistics(results)