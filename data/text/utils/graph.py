import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nllb_langs import code_mapping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process translation dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Find missing languages and add them with 0 counts
    missing_langs = set(code_mapping.values()) - set(df['nllb_code'])
    missing_df = pd.DataFrame({'nllb_code': list(missing_langs), 'total_count': [0] * len(missing_langs)})
    
    # Combine all data
    all_data = pd.concat([df, missing_df], ignore_index=True)
    all_counts = all_data.set_index('nllb_code')['total_count'].sort_values(ascending=False)
    
    logger.info(f"Total languages: {len(all_counts)}, Missing: {len(missing_langs)}")
    
    # Create simple bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(all_counts)), all_counts.values, color='steelblue', alpha=0.7)
    plt.title('Language Distribution by Total Count', fontsize=14)
    plt.xlabel('Languages (sorted by count)')
    plt.ylabel('Total Count')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add some stats as text
    stats_text = f'Total: {len(all_counts)} | Non-zero: {(all_counts > 0).sum()} | Zero: {(all_counts == 0).sum()}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / "language_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✅ Done!")
    return 0

if __name__ == "__main__":
    exit(main())