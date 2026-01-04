#!/bin/bash

# Script to compress the entire repository, including only files tracked by git

set -e  # Exit on error

# Configuration
REPO_NAME=$(basename "$(pwd)")
OUTPUT_FILE="${REPO_NAME}.tar.gz"
DATA_OUTPUT_FILE="${REPO_NAME}-data.tar.gz"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository."
    exit 1
fi

echo "Compressing repository: $REPO_NAME"
echo ""

# Get all tracked files from git and create archive
# Using git ls-files -z to get null-terminated file list (excludes .gitignore and untracked files)
echo "1. Creating full repository archive..."
echo "   Output file: $OUTPUT_FILE"
echo "   Collecting tracked files..."
git ls-files -z | tar --null -T - -czf "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    # Get file size for display
    if command -v du > /dev/null 2>&1; then
        SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "   ✓ Success! Archive created: $OUTPUT_FILE"
        echo "     Size: $SIZE"
    else
        echo "   ✓ Success! Archive created: $OUTPUT_FILE"
    fi
else
    echo "   ✗ Failed to create archive"
    exit 1
fi

echo ""

# Create archive for data directory only
if [ -d "data" ]; then
    echo "2. Creating data directory archive..."
    echo "   Output file: $DATA_OUTPUT_FILE"
    echo "   Collecting tracked files in data/..."
    
    # Get only tracked files in the data directory
    git ls-files -z data/ | tar --null -T - -czf "$DATA_OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        # Get file size for display
        if command -v du > /dev/null 2>&1; then
            SIZE=$(du -h "$DATA_OUTPUT_FILE" | cut -f1)
            echo "   ✓ Success! Archive created: $DATA_OUTPUT_FILE"
            echo "     Size: $SIZE"
        else
            echo "   ✓ Success! Archive created: $DATA_OUTPUT_FILE"
        fi
    else
        echo "   ✗ Failed to create data archive"
        exit 1
    fi
else
    echo "2. Skipping data directory archive (data/ not found)"
fi

echo "-----------------------------------"
echo "Done! All archives created."

