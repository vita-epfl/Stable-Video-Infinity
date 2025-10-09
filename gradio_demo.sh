#!/bin/bash

# Stable Video Infinity - Demo Launch Script
# Supports automatic mode switching between Film and Shot modes

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Print colored banner
print_banner() {
    echo -e "${CYAN}"
    echo "======================================================================"
    echo "🎬                STABLE VIDEO INFINITY DEMO                      🎬"
    echo "                   Supports Film/Shot Mode Switching                 "
    echo "======================================================================"
    echo -e "${NC}"
}

# Print usage instructions
print_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [options]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  -h, --help          Show this help message"
    echo "  -p, --port PORT     Specify port (default: 7860)"
    echo "  -s, --share         Enable Gradio public sharing"
    echo "  --host HOST         Specify host address (default: 0.0.0.0)"
    echo "  --dit PATH          Specify DIT model root directory"
    echo ""
    echo -e "${YELLOW}Preset Configurations:${NC}"
    echo "  --film              Start with SVI-Film mode by default"
    echo "  --shot              Start with SVI-Shot mode by default"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0                           # Default startup (Film mode)"
    echo "  $0 --shot --port 8080        # Use Shot mode, port 8080"
    echo "  $0 --film --share            # Use Film mode, enable public sharing"
}

# Check dependencies
check_dependencies() {
    echo -e "${BLUE}🔍 Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Error: Python not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Dependencies check completed${NC}"
}

# Check model files
check_models() {
    local dit_root="$1"
    
    echo -e "${BLUE}🔍 Checking model files...${NC}"
    
    # Check DIT model directory
    if [ ! -d "$dit_root" ]; then
        echo -e "${RED}❌ Error: DIT model directory does not exist: $dit_root${NC}"
        return 1
    fi
    
    # Check LoRA model files
    local lora_files=(
        "weights/Stable-Video-Infinity/version-1.0/svi-film.safetensors"
        "weights/Stable-Video-Infinity/version-1.0/svi-shot.safetensors"
    )
    
    for lora_file in "${lora_files[@]}"; do
        if [ -f "$lora_file" ]; then
            echo -e "${GREEN}✅ Found: $(basename "$lora_file")${NC}"
        else
            echo -e "${YELLOW}⚠️  Warning: LoRA file does not exist: $lora_file${NC}"
        fi
    done
    
    echo -e "${GREEN}✅ Model check completed${NC}"
}

# Show system information
show_system_info() {
    echo -e "${BLUE}💻 System Information:${NC}"
    echo "  Operating System: $(uname -s)"
    echo "  Python Version: $(python --version 2>&1)"
    echo "  Working Directory: $SCRIPT_DIR"
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        echo "  GPU: $gpu_info"
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo "  GPU Memory: ${gpu_memory}MB"
    else
        echo "  GPU: No NVIDIA GPU detected"
    fi
    echo ""
}

# Default configuration
DEFAULT_DIT_ROOT="./weights/Wan2.1-I2V-14B-480P/"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="7860"
DEFAULT_ARCH="lora"
DEFAULT_ALPHA="1.0"

# Current configuration
DIT_ROOT="$DEFAULT_DIT_ROOT"
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
ARCH="$DEFAULT_ARCH"
ALPHA="$DEFAULT_ALPHA"
SHARE_FLAG=""
PRESET=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_banner
            print_usage
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -s|--share)
            SHARE_FLAG="--share"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --dit)
            DIT_ROOT="$2"
            shift 2
            ;;
        --film)
            PRESET="film"
            shift
            ;;
        --shot)
            PRESET="shot"
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Main function
main() {
    print_banner
    
    # Show configuration information
    echo -e "${PURPLE}🔧 Launch Configuration:${NC}"
    echo "  Host Address: $HOST"
    echo "  Port: $PORT"
    echo "  DIT Model Root: $DIT_ROOT"
    echo "  Architecture: $ARCH"
    echo "  LoRA Alpha: $ALPHA"
    if [ -n "$PRESET" ]; then
        echo "  Preset Mode: $PRESET"
    else
        echo "  Default Mode: Film (switchable in interface)"
    fi
    if [ -n "$SHARE_FLAG" ]; then
        echo "  Public Sharing: Enabled"
    fi
    echo ""
    
    # System information
    show_system_info
    
    # Check dependencies
    check_dependencies
    echo ""
    
    # Check models
    check_models "$DIT_ROOT"
    echo ""
    
    # Create necessary directories
    echo -e "${BLUE}📁 Creating necessary directories...${NC}"
    mkdir -p videos
    mkdir -p logs
    echo -e "${GREEN}✅ Directory creation completed${NC}"
    echo ""
    
    # Build startup command - use Film as default mode
    local cmd="python gradio_demo.py"
    cmd="$cmd --dit_root \"$DIT_ROOT\""
    cmd="$cmd --extra_module_root \"weights/Stable-Video-Infinity/version-1.0/svi-film.safetensors\""
    cmd="$cmd --host $HOST"
    cmd="$cmd --port $PORT"
    cmd="$cmd --train_architecture $ARCH"
    cmd="$cmd --lora_alpha $ALPHA"
    if [ -n "$SHARE_FLAG" ]; then
        cmd="$cmd $SHARE_FLAG"
    fi
    
    # Show startup information
    echo -e "${GREEN}🚀 Starting Stable Video Infinity Demo...${NC}"
    echo -e "${CYAN}Command: $cmd${NC}"
    echo ""
    echo -e "${YELLOW}📱 Access URLs:${NC}"
    if [ "$HOST" = "0.0.0.0" ]; then
        echo "  Local access: http://localhost:$PORT"
        echo "  LAN access: http://$(hostname -I | awk '{print $1}'):$PORT"
    else
        echo "  Access URL: http://$HOST:$PORT"
    fi
    echo ""
    echo -e "${BLUE}💡 Usage Tips:${NC}"
    echo "  - Interface supports Film and Shot mode switching"
    echo "  - Film mode: Suitable for cinematic narratives and storylines"
    echo "  - Shot mode: Suitable for camera movements and shooting effects"
    echo "  - Each mode has corresponding demo samples"
    echo "  - Use Ctrl+C to stop the service"
    echo "  - Generated videos are saved in ./videos/ directory"
    echo ""
    echo -e "${CYAN}======================================================================"
    echo "                           Starting...                               "
    echo "======================================================================${NC}"
    echo ""
    
    # Execute startup command
    eval $cmd
}

# Signal handling
trap 'echo -e "\n${YELLOW}🛑 Stopping service...${NC}"; exit 0' INT TERM

# Run main function
main "$@"