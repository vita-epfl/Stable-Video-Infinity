import os
from datetime import datetime
import shutil
from pathlib import Path
import yaml

def update_experiment_path(args, short=False):
    path_components = []
    mode = "ft" if args.pretrained_lora_path else "scratch"
    path_components.append(f"{args.train_architecture}")
    if args.train_architecture == "lora":
        path_components.append(f"{args.lora_rank}")
    
    if not short:
        path_components.append(f"pose_cfg-{args.pose_cfg}")
        path_components.append(f"mouth_cfg-{args.mouth_cfg}")
        path_components.append(f"pose_relax-{args.pose_relax}-f{args.pose_relax_num}-{mode}")

    experiment_name = "_".join(path_components)
    experiment_name = "{}-".format(args.exp_prefix) + experiment_name if args.exp_prefix else experiment_name

    full_path = os.path.join(args.output_path, experiment_name)
    print(f"Experiment path: {full_path}")
    os.makedirs(full_path, exist_ok=True)

    args.output_path = full_path
    return args


def print_args(args):
    print("=" * 80)
    print("CONFIGURATION PARAMETERS:")
    print("=" * 80)
    
    args_dict = vars(args)
    max_key_length = max(len(key) for key in args_dict.keys())
    
    for key in sorted(args_dict.keys()):
        value = args_dict[key]
        print(f"  {key.ljust(max_key_length)} : {value}")
    
    print("=" * 80)
    print(f"Total number of cfg parameters: {len(args_dict)}")
    print("=" * 80)


def save_args_to_yaml(args, output_path):
    """
    Save all arguments to a YAML file in the output directory.
    Only saves from the main process in distributed training to avoid conflicts.
    
    Args:
        args: Parsed arguments from argparse
        output_path: Directory where to save the args.yaml file
    """
    # In distributed training, only save from rank 0 (main process)
    try:
        # Try to get the current process rank
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return  # Skip saving from non-main processes
    except:
        # If distributed training is not set up, continue with saving
        pass
    
    # Check environment variable for local rank (common in distributed setups)
    local_rank = os.environ.get('LOCAL_RANK', '0')
    global_rank = os.environ.get('RANK', '0')
    if local_rank != '0' or global_rank != '0':
        return  # Skip saving from non-main processes
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Save to YAML file with file locking to prevent conflicts
    yaml_path = os.path.join(output_path, 'args.yaml')
    lock_path = yaml_path + '.lock'
    
    try:
        # Simple file-based locking mechanism
        if os.path.exists(lock_path):
            # Another process is writing, skip
            return
            
        # Create lock file
        with open(lock_path, 'w') as lock_file:
            lock_file.write(str(os.getpid()))
        
        # Save args to YAML
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(args_dict, f, default_flow_style=False, sort_keys=True, indent=2)
        print(f"Arguments saved to: {yaml_path}")
        
        # Remove lock file
        if os.path.exists(lock_path):
            os.remove(lock_path)
            
    except Exception as e:
        print(f"Warning: Failed to save arguments to {yaml_path}: {e}")
        # Clean up lock file if it exists
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except:
                pass
