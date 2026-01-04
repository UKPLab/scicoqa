#!/bin/bash

# Script to extract all out/ directory archives to their correct locations
# Archives are stored in the out/ directory (e.g., out/data_collection.tar.gz)
# and will be extracted to restore the out/ directory structure

set -e  # Exit on error

ARCHIVE_DIR="out"

echo "Extracting archives from '$ARCHIVE_DIR/' to restore directory structure..."
echo "Note: Archives are located in '$ARCHIVE_DIR/' and will be extracted there."
echo ""

# Check if archive directory exists
if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "Error: Directory '$ARCHIVE_DIR' not found."
    exit 1
fi

# Create out directory structure if it doesn't exist
mkdir -p out

# Function to extract archive
extract_archive() {
    local archive="$1"
    local description="$2"
    
    if [ ! -f "$archive" ]; then
        echo "  ⚠ Skipping '$archive' (not found)"
        return
    fi
    
    echo "  Extracting '$archive' ($description)..."
    
    # Extract archive to current directory (archives contain full paths from project root)
    tar -xzf "$archive"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully extracted '$description'"
    else
        echo "  ✗ Failed to extract '$archive'"
        return 1
    fi
}

# Extract data_collection archive
extract_archive "$ARCHIVE_DIR/data_collection.tar.gz" "data_collection"

# Extract inference archives
# Real data (contains both full and code_only)
extract_archive "$ARCHIVE_DIR/inference_discrepancy_detection_real.tar.gz" "inference/discrepancy_detection/real"

# Synthetic data - code_only
extract_archive "$ARCHIVE_DIR/inference_discrepancy_detection_synthetic_code_only.tar.gz" "inference/discrepancy_detection/synthetic/code_only"

# Synthetic data - full
extract_archive "$ARCHIVE_DIR/inference_discrepancy_detection_synthetic_full.tar.gz" "inference/discrepancy_detection/synthetic/full"

# Extract any other inference_*.tar.gz files (excluding the ones we already handled)
for archive in "$ARCHIVE_DIR"/inference_*.tar.gz; do
    if [ -f "$archive" ]; then
        archive_name=$(basename "$archive")
        # Skip archives we've already processed
        case "$archive_name" in
            inference_discrepancy_detection_real.tar.gz|inference_discrepancy_detection_synthetic_code_only.tar.gz|inference_discrepancy_detection_synthetic_full.tar.gz)
                continue
                ;;
        esac
        extract_archive "$archive" "$archive_name"
    fi
done

echo "-----------------------------------"
echo "Done! All archives extracted to 'out/' directory"
echo ""
echo "Directory structure:"
echo "  out/"
echo "  ├── data_collection/"
echo "  └── inference/"
echo "      └── discrepancy_detection/"
echo "          ├── real/"
echo "          │   ├── code_only/"
echo "          │   └── full/"
echo "          └── synthetic/"
echo "              ├── code_only/"
echo "              └── full/"

