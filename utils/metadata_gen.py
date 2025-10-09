import os
import csv
import argparse
from pathlib import Path

def read_caption_file(caption_path):
    """
    Read caption file content
    
    Args:
        caif __name__ == "__main__":
    # If no command line arguments, use default settings
    if len(os.sys.argv) == 1:
        print("Running with default settings...")
        
        video_directory = "/mnt/vita/scratch/vita-students/users/wuli/data/video-gen/Hallo3_5k/videos/"
        caption_directory = "/mnt/vita/scratch/vita-students/users/wuli/data/video-gen/Hallo3_5k/captions"
        output_file = "metadata.csv"
        default_description = "A person is dancing"h: Path to caption file
        
    Returns:
        Caption text content, returns None if file doesn't exist or read fails
    """
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content
    except Exception as e:
        print(f"Warning: Failed to read caption file {caption_path}: {e}")
        return None

def generate_metadata_with_captions(video_dir, caption_dir, output_csv, default_text="A person is dancing"):
    """
    Generate metadata.csv based on caption files
    
    Args:
        video_dir: Video files directory
        caption_dir: Caption files directory
        output_csv: Output CSV file path
        default_text: Default text to use when corresponding caption is not found
    """
    
    # Check if directories exist
    if not os.path.exists(video_dir):
        print(f"Error: Video directory '{video_dir}' does not exist!")
        return
    
    if not os.path.exists(caption_dir):
        print(f"Error: Caption directory '{caption_dir}' does not exist!")
        return
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for file in os.listdir(video_dir):
        file_path = os.path.join(video_dir, file)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in video_extensions:
                video_files.append(file)
    
    # Sort files by name
    video_files.sort()
    
    print(f"Found {len(video_files)} video files")
    
    # Statistics
    found_captions = 0
    missing_captions = 0
    
    # Write to CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['file_name', 'text'])
        
        # Process each video file
        for video_file in video_files:
            # Get filename without extension
            video_name = os.path.splitext(video_file)[0]
            
            # Try multiple possible caption file extensions
            caption_extensions = ['.txt', '.caption', '.text']
            caption_text = None
            
            for ext in caption_extensions:
                caption_file = video_name + ext
                caption_path = os.path.join(caption_dir, caption_file)
                
                if os.path.exists(caption_path):
                    caption_text = read_caption_file(caption_path)
                    if caption_text:
                        found_captions += 1
                        break
            
            # If no caption file found, use default text
            if caption_text is None:
                caption_text = default_text
                missing_captions += 1
                print(f"Warning: No caption found for {video_file}, using default text")
            
            # Write CSV row
            writer.writerow([video_file, caption_text])
    
    print(f"\nSummary:")
    print(f"Total videos: {len(video_files)}")
    print(f"Found captions: {found_captions}")
    print(f"Missing captions: {missing_captions}")
    print(f"Successfully generated metadata.csv: {output_csv}")

def list_caption_files(caption_dir):
    """
    List files in caption directory for debugging
    """
    if not os.path.exists(caption_dir):
        print(f"Caption directory does not exist: {caption_dir}")
        return
    
    files = os.listdir(caption_dir)
    print(f"Found {len(files)} files in caption directory:")
    
    # Group by file extension
    extensions = {}
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext not in extensions:
            extensions[ext] = []
        extensions[ext].append(file)
    
    for ext, file_list in extensions.items():
        print(f"  {ext}: {len(file_list)} files")
        if len(file_list) <= 5:  # Show first 5 files as examples
            for file in file_list:
                print(f"    {file}")
        else:
            for file in file_list[:3]:
                print(f"    {file}")
            print(f"    ... and {len(file_list)-3} more")

def main():
    parser = argparse.ArgumentParser(description='Generate metadata.csv for video files with captions')
    parser.add_argument('--video_dir', type=str, 
                        default="/mnt/vita/scratch/vita-students/users/wuli/data/video-gen/Hallo3_1000/videos/",
                        help='Directory containing video files')
    parser.add_argument('--caption_dir', type=str,
                        default="/mnt/vita/scratch/vita-students/users/wuli/data/video-gen/Hallo3_5k/captions",
                        help='Directory containing caption files')
    parser.add_argument('--output_csv', type=str, default='metadata.csv',
                        help='Output CSV file path (default: metadata.csv)')
    parser.add_argument('--default_text', type=str, default='A person is speaking',
                        help='Default description text when caption is not found')
    parser.add_argument('--list_captions', action='store_true',
                        help='List caption files in the directory (for debugging)')
    
    args = parser.parse_args()
    
    if args.list_captions:
        print("Listing caption files...")
        list_caption_files(args.caption_dir)
        return
    
    print(f"Video directory: {args.video_dir}")
    print(f"Caption directory: {args.caption_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Default text: {args.default_text}")
    print()
    
    generate_metadata_with_captions(args.video_dir, args.caption_dir, args.output_csv, args.default_text)

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认设置
    if len(os.sys.argv) == 1:
        print("Running with default settings...")
        
        video_directory = "/mnt/vita/scratch/vita-students/users/wuli/data/video-gen/Hallo3_5k/videos/"
        caption_directory = "/mnt/vita/scratch/vita-students/users/wuli/data/video-gen/Hallo3_5k/captions"
        output_file = "metadata.csv"
        default_description = "A person is speaking"
        
        print(f"Video directory: {video_directory}")
        print(f"Caption directory: {caption_directory}")
        print(f"Output file: {output_file}")
        print(f"Default text: {default_description}")
        print()
        
        generate_metadata_with_captions(video_directory, caption_directory, output_file, default_description)
    else:
        main()