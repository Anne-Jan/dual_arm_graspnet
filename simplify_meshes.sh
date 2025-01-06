#!/bin/bash

# Check if arguments are provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Assign the provided directory to a variable
directory=$1

# Loop through each file in the directory
for file in "$directory"/*; do
    if [ -f "$file" ]; then
        # Run the first command
        ./manifold $file temp.watertight.obj
        rm $file
        
        # Run the second command
        ./simplify -i temp.watertight.obj -o $file -m -r 0.02
        
        # Optionally remove temp file
        rm temp.watertight.obj
    fi
done
