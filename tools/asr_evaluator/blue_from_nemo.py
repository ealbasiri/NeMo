import json
import argparse
from collections import defaultdict
from typing import Dict, List
from torchmetrics.text import SacreBLEUScore
import torch

def load_manifest(manifest_path: str) -> List[dict]:
    """Load and parse the manifest file."""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def group_by_lang_pair(manifest_data: List[dict]) -> Dict[str, List[dict]]:
    """Group manifest entries by language pair."""
    lang_pairs = defaultdict(list)
    for entry in manifest_data:
        lang_pair = f"{entry['source_lang']}-{entry['target_lang']}"
        lang_pairs[lang_pair].append(entry)
    return lang_pairs

def calculate_bleu_scores(grouped_data: Dict[str, List[dict]]) -> Dict[str, float]:
    """Calculate BLEU score for each language pair."""
    bleu_scores = {}
    
    for lang_pair, entries in grouped_data.items():
        # Initialize BLEU metric
        bleu_metric = SacreBLEUScore(
            tokenize='13a',  # Using standard BLEU tokenization
            lowercase=True   # Convert to lowercase for comparison
        )
        
        # Prepare predictions and references
        predictions = [entry['pred_text'] for entry in entries]
        references = [[entry['text']] for entry in entries]  # Double brackets as BLEU expects multiple references
        
        # Update BLEU metric
        bleu_metric.update(predictions, references)
        
        # Compute final score
        score = bleu_metric.compute().item()
        bleu_scores[lang_pair] = score
        
    return bleu_scores

def main():
    parser = argparse.ArgumentParser(description='Calculate BLEU scores for language pairs in a prediction manifest')
    parser.add_argument('manifest_path', type=str, help='Path to the prediction manifest file')
    parser.add_argument('--output', type=str, default='bleu_scores.json', help='Path to save the BLEU scores')
    
    args = parser.parse_args()
    
    # Load and process manifest
    print(f"Loading manifest from {args.manifest_path}")
    manifest_data = load_manifest(args.manifest_path)
    
    # Group by language pairs
    print("Grouping data by language pairs")
    grouped_data = group_by_lang_pair(manifest_data)
    
    # Calculate BLEU scores
    print("Calculating BLEU scores")
    bleu_scores = calculate_bleu_scores(grouped_data)
    
    # Print results
    print("\nBLEU Scores by Language Pair:")
    print("-" * 40)
    for lang_pair, score in sorted(bleu_scores.items()):
        print(f"{lang_pair}: {score:.4f}")
    
    # Save results to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(bleu_scores, f, indent=2)
    print(f"\nScores saved to {args.output}")
    
    # Calculate and print average BLEU score
    avg_bleu = sum(bleu_scores.values()) / len(bleu_scores)
    print(f"\nAverage BLEU score across all language pairs: {avg_bleu:.4f}")

if __name__ == "__main__":
    main()