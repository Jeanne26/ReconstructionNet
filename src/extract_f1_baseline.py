"""
Extract F1-scores from baseline log text files.
Computes mean and std for each model type (mnist, octmnist, diabetes).
"""

import os
import glob
import re
import numpy as np
from collections import defaultdict

LOGS_DIR = '../logs_sh_baseline/'

def extract_f1_from_log(log_file):
    """Extract weighted avg F1-score from a classification report in log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for weighted avg line
        # Format: "weighted avg       0.95      0.89      0.91     10000"
        pattern = r'weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)'
        match = re.search(pattern, content)
        
        if match:
            return float(match.group(1))
        
    except Exception as e:
        print(f"  Error reading {log_file}: {e}")
        return None
    
    return None


def main():
    # Find all log files
    log_files = sorted(glob.glob(os.path.join(LOGS_DIR, '*.txt')))
    
    if not log_files:
        print(f"No log files found in {LOGS_DIR}")
        return
    
    # Group by model type (mnist, octmnist, diabetes, isic)
    model_f1_scores = defaultdict(list)
    
    print("Extracting F1-scores from baseline logs...\n")
    
    for log_file in log_files:
        log_name = os.path.basename(log_file).replace('.txt', '')
        
        # Extract model type from filename
        # e.g., "logs_baseline_diabetes_seed_0" -> "diabetes"
        # Pattern: logs_baseline_{model}_seed_{num}
        match = re.search(r'logs_baseline_(\w+)_seed_(\d+)', log_name)
        if match:
            model_type = match.group(1)
            seed = match.group(2)
        else:
            model_type = "unknown"
            seed = "?"
        
        f1_score = extract_f1_from_log(log_file)
        
        if f1_score is not None:
            model_f1_scores[model_type].append(f1_score)
            print(f"  {model_type:10s} seed {seed:2s}  →  F1 = {f1_score:.4f}")
        else:
            print(f"  {model_type:10s} seed {seed:2s}  →  No F1-score found")
    
    # Compute mean and std for each model type
    print("\n" + "="*60)
    print("Summary Statistics (Mean ± Std)")
    print("="*60)
    
    for model_type in sorted(model_f1_scores.keys()):
        f1_values = np.array(model_f1_scores[model_type])
        mean_f1 = f1_values.mean()
        std_f1 = f1_values.std()
        n = len(f1_values)
        
        print(f"{model_type:15s} (n={n:2d}):  {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"                     Values: {[f'{v:.4f}' for v in f1_values]}")
    
    print("="*60)


if __name__ == "__main__":
    main()
