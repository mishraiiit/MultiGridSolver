#!/bin/bash

# Script to concatenate specified source files from a directory into a single output file.

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <source_directory> [output_file]"
    echo "Example: $0 src/GPU_CUDAC++ combined_gpu_sources.txt"
    exit 1
fi

SOURCE_DIR="$1"
OUTPUT_FILE="${2:-combined_sources.txt}" # Default output filename is combined_sources.txt

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found."
    exit 1
fi

# Clear the output file if it exists, or create it
> "$OUTPUT_FILE"

echo "Combining files from '$SOURCE_DIR' into '$OUTPUT_FILE' (skipping files with 'test' in their name)..."

# Find and process .cu, .cpp, .h, .cuh, and .txt files, excluding those with 'test' in the name (case-insensitive)
# Using find with -print0 and piping directly to while read -r -d $'\0'
find "$SOURCE_DIR" -maxdepth 1 \( -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.cuh' -o -name '*.txt' \) -not -iname '*test*' -print0 | while IFS= read -r -d $'\0' file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing $filename..." # This is the original echo
        echo "==== Contents of $filename ====" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE" # Add a newline for separation
        echo "" >> "$OUTPUT_FILE" # Add another newline for better readability
    else
        : # No-op, or a more subtle log if needed
    fi
done

echo "Concatenation complete. Output is in '$OUTPUT_FILE'." 