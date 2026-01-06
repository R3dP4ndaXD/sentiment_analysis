#!/usr/bin/env python3
"""
Offline augmentation script for heavy augmentation techniques.

Creates an expanded and balanced training dataset by:
1. Augmenting the minority class more heavily to achieve balance
2. Pre-computing expensive augmentations (back-translation, contextual) and saving to disk

Usage:
    python -m src.data.offline_augmentation \
        --input data/processed/train.csv \
        --output data/augmented/train_balanced.csv \
        --augment back_translate contextual \
        --target_ratio 1.0 \
        --device cuda

    # Or generate multiple augmented versions per sample
    python -m src.data.offline_augmentation \
        --input data/processed/train.csv \
        --output data/augmented/train_expanded.csv \
        --augment back_translate \
        --augment_per_sample 2 \
        --balance
"""
import argparse
import random
from pathlib import Path
from typing import List, Optional, Callable
from tqdm import tqdm
import pandas as pd

# Augmentation imports
from src.data.augmentations import (
    back_translate,
    contextual_word_replacement,
    contextual_insert,
    synonym_replacement_wordnet,
    random_swap,
    random_delete,
)
from src.preprocessing.text import tokenize, detokenize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline data augmentation with balancing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument(
        "--augment", type=str, nargs="+", 
        choices=["back_translate", "contextual", "contextual_insert", "synonym", "swap", "delete"],
        default=["back_translate"],
        help="Augmentation techniques to apply (split evenly across techniques)"
    )
    parser.add_argument(
        "--augment_per_sample", type=int, default=1,
        help="Number of augmented versions to generate per sample"
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Balance classes by augmenting minority class more"
    )
    parser.add_argument(
        "--target_ratio", type=float, default=1.0,
        help="Target ratio of minority to majority class (1.0 = balanced)"
    )
    parser.add_argument(
        "--minority_only", action="store_true",
        help="Only augment the minority class"
    )
    parser.add_argument(
        "--include_original", action="store_true", default=True,
        help="Include original samples in output"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--text_col", type=str, default="text", help="Text column name"
    )
    parser.add_argument(
        "--label_col", type=str, default="label", help="Label column name"
    )
    return parser.parse_args()


def get_augmentation_fn(
    aug_name: str,
    device: str = "cpu",
) -> Callable[[List[str]], List[str]]:
    """Get augmentation function by name."""
    if aug_name == "back_translate":
        return lambda tokens: back_translate(tokens, device=device)
    elif aug_name == "contextual":
        return lambda tokens: contextual_word_replacement(
            tokens, n_replacements=2, device=device
        )
    elif aug_name == "contextual_insert":
        return lambda tokens: contextual_insert(
            tokens, n_inserts=2, device=device
        )
    elif aug_name == "synonym":
        return lambda tokens: synonym_replacement_wordnet(tokens, n_replacements=2)
    elif aug_name == "swap":
        return lambda tokens: random_swap(tokens, n_swaps=2)
    elif aug_name == "delete":
        return lambda tokens: random_delete(tokens, p=0.1)
    else:
        raise ValueError(f"Unknown augmentation: {aug_name}")


def augment_text(
    text: str,
    augment_fns: List[Callable],
    tokenizer: Callable = tokenize,
) -> str:
    """Apply augmentation pipeline to text.
    
    Args:
        text: Input text
        augment_fns: List of augmentation functions to apply in sequence
        tokenizer: Tokenization function
    
    Returns:
        Augmented text
    """
    tokens = tokenizer(text)
    if not tokens:
        return text
    
    # Apply each augmentation
    for aug_fn in augment_fns:
        try:
            tokens = aug_fn(tokens)
            if not tokens:  # If augmentation returned empty, use original
                tokens = tokenizer(text)
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
            continue
    
    return detokenize(tokens)


def compute_augmentation_counts(
    df: pd.DataFrame,
    label_col: str,
    augment_per_sample: int,
    balance: bool,
    target_ratio: float,
    minority_only: bool,
) -> dict:
    """Compute how many augmentations to generate per class.
    
    Returns:
        Dict mapping label -> number of augmentations to generate
    """
    class_counts = df[label_col].value_counts().to_dict()
    labels = list(class_counts.keys())
    
    majority_label = max(class_counts, key=class_counts.get)
    minority_label = min(class_counts, key=class_counts.get)
    
    majority_count = class_counts[majority_label]
    minority_count = class_counts[minority_label]
    
    print(f"Class distribution:")
    print(f"  Majority (label={majority_label}): {majority_count} ({majority_count/len(df)*100:.1f}%)")
    print(f"  Minority (label={minority_label}): {minority_count} ({minority_count/len(df)*100:.1f}%)")
    
    aug_counts = {}
    
    if balance:
        # Calculate how many samples needed to balance
        target_minority = int(majority_count * target_ratio)
        minority_augments_needed = max(0, target_minority - minority_count)
        
        # Augment minority class
        aug_counts[minority_label] = minority_augments_needed
        
        # Optionally augment majority class too
        if not minority_only:
            # Add same number of augments per sample as minority would get on average
            avg_aug_per_minority = minority_augments_needed / minority_count if minority_count > 0 else 0
            aug_counts[majority_label] = int(majority_count * min(avg_aug_per_minority, augment_per_sample))
        else:
            aug_counts[majority_label] = 0
    else:
        # Simple augmentation: same number per sample
        for label in labels:
            if minority_only and label == majority_label:
                aug_counts[label] = 0
            else:
                aug_counts[label] = class_counts[label] * augment_per_sample
    
    return aug_counts, minority_label, majority_label


def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    df = df.dropna(subset=[args.text_col])
    print(f"Loaded {len(df)} samples")
    
    # Setup augmentation functions
    augment_fns = [get_augmentation_fn(aug, args.device) for aug in args.augment]
    print(f"Augmentation pipeline: {args.augment}")
    
    # Compute augmentation counts
    aug_counts, minority_label, majority_label = compute_augmentation_counts(
        df=df,
        label_col=args.label_col,
        augment_per_sample=args.augment_per_sample,
        balance=args.balance,
        target_ratio=args.target_ratio,
        minority_only=args.minority_only,
    )
    
    print(f"\nAugmentation plan:")
    for label, count in aug_counts.items():
        print(f"  Label {label}: generate {count} augmented samples")
    
    # Prepare output data
    output_rows = []
    
    # Include original samples
    if args.include_original:
        for _, row in df.iterrows():
            output_rows.append({
                args.text_col: row[args.text_col],
                args.label_col: row[args.label_col],
                "augmented": False,
                "aug_type": "original",
            })
    
    # Generate augmented samples
    for label in aug_counts:
        count = aug_counts[label]
        if count == 0:
            continue
        
        # Get samples of this class
        class_df = df[df[args.label_col] == label]
        class_texts = class_df[args.text_col].tolist()
        
        # Split count evenly across augmentation techniques
        n_augs = len(args.augment)
        base_count = count // n_augs
        remainder = count % n_augs
        
        print(f"\nGenerating {count} augmented samples for label {label}...")
        print(f"  Splitting across {n_augs} techniques: {args.augment}")
        
        for aug_idx, aug_name in enumerate(args.augment):
            # Distribute remainder to first few augmenters
            aug_count = base_count + (1 if aug_idx < remainder else 0)
            if aug_count == 0:
                continue
            
            aug_fn = augment_fns[aug_idx]
            
            print(f"  {aug_name}: {aug_count} samples")
            
            # Generate augmentations for this technique
            generated = 0
            attempts = 0
            max_attempts = aug_count * 3  # Prevent infinite loops
            
            pbar = tqdm(total=aug_count, desc=f"Label {label} [{aug_name}]")
            
            while generated < aug_count and attempts < max_attempts:
                # Sample a random text from this class
                text = random.choice(class_texts)
                
                try:
                    # Apply single augmentation (not chained)
                    aug_text = augment_text(text, [aug_fn])
                    
                    # Check if augmentation produced something different
                    if aug_text and aug_text != text and len(aug_text) > 10:
                        output_rows.append({
                            args.text_col: aug_text,
                            args.label_col: label,
                            "augmented": True,
                            "aug_type": aug_name,
                        })
                        generated += 1
                        pbar.update(1)
                except Exception as e:
                    print(f"Warning: {aug_name} failed: {e}")
                
                attempts += 1
            
            pbar.close()
            
            if generated < aug_count:
                print(f"Warning: Only generated {generated}/{aug_count} samples for {aug_name}")
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_rows)
    
    # Shuffle
    output_df = output_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # Print final statistics
    print(f"\n{'='*50}")
    print("Output Statistics:")
    print(f"{'='*50}")
    print(f"Total samples: {len(output_df)}")
    print(f"\nClass distribution:")
    for label in output_df[args.label_col].unique():
        count = (output_df[args.label_col] == label).sum()
        pct = count / len(output_df) * 100
        print(f"  Label {label}: {count} ({pct:.1f}%)")
    
    print(f"\nAugmented vs Original:")
    aug_count = output_df["augmented"].sum()
    orig_count = len(output_df) - aug_count
    print(f"  Original: {orig_count} ({orig_count/len(output_df)*100:.1f}%)")
    print(f"  Augmented: {aug_count} ({aug_count/len(output_df)*100:.1f}%)")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata columns
    output_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    # Also save a clean version without metadata (just text, label)
    clean_path = output_path.with_stem(output_path.stem + "_clean")
    output_df[[args.text_col, args.label_col]].to_csv(clean_path, index=False)
    print(f"✓ Clean version saved to {clean_path}")


if __name__ == "__main__":
    main()
