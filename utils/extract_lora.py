#!/usr/bin/env python3
"""
Extract LoRA weights from sharded safetensors checkpoint and save as individual safetensors files.
Compatible with test_svi.py's load_lora_v2 method.
"""

import os
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from collections import defaultdict
import glob
from tqdm import tqdm


def load_sharded_checkpoint(checkpoint_dir):
    """Load all shards from a checkpoint directory."""
    # Find all safetensors files
    shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*-of-*.safetensors")))
    
    if not shard_files:
        raise FileNotFoundError(f"No safetensors shards found in {checkpoint_dir}")
    
    print(f"Found {len(shard_files)} safetensors shards")
    
    # Load all weights from shards
    all_weights = {}
    for shard_file in tqdm(shard_files, desc="Loading shards"):
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)
    
    print(f"Loaded {len(all_weights)} tensors from checkpoint")
    return all_weights


def extract_lora_weights(all_weights, verbose=False):
    """Extract LoRA weights and dwpose_embedding from checkpoint."""
    lora_weights = {}
    
    # Patterns to identify LoRA weights and dwpose_embedding
    lora_patterns = ['lora_A', 'lora_B', 'lora_', 'alpha']
    dwpose_patterns = ['dwpose_embedding']
    
    for key, value in all_weights.items():
        # Check if this is a LoRA weight or dwpose_embedding
        is_lora = any(pattern in key for pattern in lora_patterns)
        is_dwpose = any(pattern in key for pattern in dwpose_patterns)
        
        if is_lora or is_dwpose:
            # Remove common prefixes that might interfere with loading
            clean_key = key
            prefixes_to_remove = ['model.', 'module.', '_orig_mod.']
            for prefix in prefixes_to_remove:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            
            lora_weights[clean_key] = value
            
            if verbose:
                print(f"Extracted: {key} -> {clean_key} (shape: {value.shape})")
    
    # Count LoRA and dwpose weights separately for reporting
    num_lora = sum(1 for k in lora_weights.keys() if any(p in k for p in ['lora_A', 'lora_B', 'lora_', 'alpha']))
    num_dwpose = sum(1 for k in lora_weights.keys() if 'dwpose_embedding' in k)
    
    print(f"Extracted {num_lora} LoRA weights")
    if num_dwpose > 0:
        print(f"Extracted {num_dwpose} dwpose_embedding weights")
    print(f"Total extracted weights: {len(lora_weights)}")
    
    return lora_weights


def save_lora_weights(lora_weights, output_dir, num_shards=1):
    """Save LoRA weights as safetensors files."""
    os.makedirs(output_dir, exist_ok=True)
    
    if num_shards == 1:
        # Save as single file
        output_path = os.path.join(output_dir, "lora_weights.safetensors")
        save_file(lora_weights, output_path)
        print(f"Saved LoRA weights to: {output_path}")
        return [output_path]
    else:
        # Split into multiple shards
        keys = list(lora_weights.keys())
        shard_size = (len(keys) + num_shards - 1) // num_shards
        
        output_paths = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(keys))
            
            shard_keys = keys[start_idx:end_idx]
            shard_weights = {k: lora_weights[k] for k in shard_keys}
            
            output_path = os.path.join(output_dir, f"lora_weights_{i+1:02d}_of_{num_shards:02d}.safetensors")
            save_file(shard_weights, output_path)
            output_paths.append(output_path)
            print(f"Saved shard {i+1}/{num_shards} to: {output_path}")
        
        return output_paths


def print_lora_stats(lora_weights):
    """Print statistics about extracted LoRA weights and dwpose_embedding."""
    print("\n" + "="*80)
    print("EXTRACTED WEIGHTS STATISTICS")
    print("="*80)
    
    # Separate LoRA and dwpose weights
    lora_only = {k: v for k, v in lora_weights.items() if 'dwpose_embedding' not in k}
    dwpose_only = {k: v for k, v in lora_weights.items() if 'dwpose_embedding' in k}
    
    # Group by layer/module
    layer_groups = defaultdict(list)
    for key in lora_only.keys():
        # Extract layer name (before the lora_A/lora_B part)
        parts = key.split('.')
        layer_name = '.'.join(parts[:-1]) if len(parts) > 1 else parts[0]
        layer_groups[layer_name].append(key)
    
    print(f"Total parameters: {len(lora_weights)}")
    print(f"  - LoRA parameters: {len(lora_only)}")
    print(f"  - dwpose_embedding parameters: {len(dwpose_only)}")
    print(f"Number of layers with LoRA: {len(layer_groups)}")
    
    # Calculate total parameters
    total_params = sum(w.numel() for w in lora_weights.values())
    lora_params = sum(w.numel() for w in lora_only.values())
    dwpose_params = sum(w.numel() for w in dwpose_only.values())
    
    print(f"Total trainable parameters: {total_params:,}")
    print(f"  - LoRA parameters: {lora_params:,}")
    print(f"  - dwpose_embedding parameters: {dwpose_params:,}")
    print(f"Total size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if dwpose_only:
        print(f"\ndwpose_embedding weights:")
        for key in dwpose_only.keys():
            shape = dwpose_only[key].shape
            print(f"  {key}: shape={shape}")
    
    print(f"\nSample layers with LoRA:")
    for i, (layer_name, keys) in enumerate(list(layer_groups.items())[:10]):
        print(f"  {layer_name}: {len(keys)} weights")
    
    if len(layer_groups) > 10:
        print(f"  ... and {len(layer_groups) - 10} more layers")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract LoRA weights from sharded checkpoint")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to directory containing sharded safetensors files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted LoRA weights"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of output shards (default: 1 for single file)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about extracted weights"
    )
    
    args = parser.parse_args()
    
    print(f"\nExtracting LoRA weights from: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of output shards: {args.num_shards}\n")
    
    # Load checkpoint
    all_weights = load_sharded_checkpoint(args.checkpoint_dir)
    
    # Extract LoRA weights
    lora_weights = extract_lora_weights(all_weights, verbose=args.verbose)
    
    if not lora_weights:
        print("ERROR: No LoRA weights found in checkpoint!")
        return
    
    # Print statistics
    print_lora_stats(lora_weights)
    
    # Save LoRA weights
    output_paths = save_lora_weights(lora_weights, args.output_dir, args.num_shards)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Saved {len(output_paths)} file(s) to: {args.output_dir}")
    print("\nTo use with test_svi.py, set:")
    print(f"  --extra_module_root {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
