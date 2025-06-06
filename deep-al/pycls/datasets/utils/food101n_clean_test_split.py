import pandas as pd
import numpy as np

def create_clean_test_split(verified_train_path, verified_val_path, output_dir):
    """
    Creates new train/test splits where test set contains only clean samples.
    
    Args:
        verified_train_path: Path to original verified_train.tsv
        verified_val_path: Path to original verified_val.tsv
        output_dir: Directory to save new tsv files
    """
    # Load verification files
    train_df = pd.read_csv(verified_train_path, sep='\t', header=0, names=['path', 'verification'])
    val_df = pd.read_csv(verified_val_path, sep='\t', header=0, names=['path', 'verification'])
    
    # Get sizes for reference
    orig_train_size = len(train_df)
    orig_val_size = len(val_df)
    
    print(f"Original sizes - Train: {orig_train_size}, Val: {orig_val_size}")
    print(f"Original noise rates - Train: {(train_df.verification==0).mean():.3f}, Val: {(val_df.verification==0).mean():.3f}")
    
    # Find noisy samples in val set
    noisy_val_samples = val_df[val_df.verification == 0]
    num_noisy_val = len(noisy_val_samples)
    
    # Get equivalent number of clean samples from train set
    clean_train_samples = train_df[train_df.verification == 1]
    selected_clean_train = clean_train_samples.sample(n=num_noisy_val, random_state=42)
    
    # Create new train set:
    # Start with original train set
    # Remove selected clean samples
    # Add noisy samples from val
    new_train_df = pd.concat([
        train_df[~train_df.index.isin(selected_clean_train.index)],  # Original train minus selected clean
        noisy_val_samples  # Add noisy samples from val
    ]).reset_index(drop=True)
    
    # Create new val set:
    # Start with original val set
    # Remove noisy samples 
    # Add selected clean samples from train
    new_val_df = pd.concat([
        val_df[val_df.verification == 1],  # Keep only clean samples from original val
        selected_clean_train  # Add clean samples from train
    ]).reset_index(drop=True)
    
    print("\nNew sizes - Train: {}, Val: {}".format(len(new_train_df), len(new_val_df)))
    print("New noise rates - Train: {:.3f}, Val: {:.3f}".format(
        (new_train_df.verification==0).mean(), 
        (new_val_df.verification==0).mean()
    ))
    
    # Save new splits
    new_train_path = f"{output_dir}/verified_train_clean_test.tsv"
    new_val_path = f"{output_dir}/verified_val_clean_test.tsv"
    
    new_train_df.to_csv(new_train_path, sep='\t', header=False, index=False)
    new_val_df.to_csv(new_val_path, sep='\t', header=False, index=False)
    
    return new_train_path, new_val_path

if __name__ == "__main__":
    # Example usage:
    base_dir = "/cs/labs/daphna/nettashaf/data/Food-101N/meta/"
    verified_train_path = f"{base_dir}/verified_train.tsv"
    verified_val_path = f"{base_dir}/verified_val.tsv"
    output_dir = base_dir
    
    new_train_path, new_val_path = create_clean_test_split(
        verified_train_path, 
        verified_val_path,
        output_dir
    )