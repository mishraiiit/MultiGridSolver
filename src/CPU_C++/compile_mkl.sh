#!/bin/bash

# Check if source file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <source_file.cpp> [output_name]"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (two levels up from script location)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Set source file and output name
SOURCE_FILE=$1
OUTPUT_NAME=${2:-${SOURCE_FILE%.cpp}}  # Use second argument or derive from source file

# Compile with Intel MKL
if g++ -std=c++11 -msse4 -O3 \
    $SOURCE_FILE \
    -o $OUTPUT_NAME \
    -L/usr/lib/x86_64-linux-gnu \
    -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
    -lpthread -lm -ldl \
    -DMKL_ILP64 -m64 \
    -I/usr/include/mkl \
    -I${PROJECT_ROOT}/lib 2>&1; then
    echo "Compilation successful. Output: $OUTPUT_NAME"
else
    echo "Compilation failed with error:"
    g++ -std=c++11 -msse4 -O3 \
        $SOURCE_FILE \
        -o $OUTPUT_NAME \
        -L/usr/lib/x86_64-linux-gnu \
        -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
        -lpthread -lm -ldl \
        -DMKL_ILP64 -m64 \
        -I/usr/include/mkl \
        -I${PROJECT_ROOT}/lib 2>&1
    exit 1
fi
