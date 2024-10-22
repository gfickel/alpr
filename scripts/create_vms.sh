#!/bin/bash

# Check if the number of instances is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_instances>"
    exit 1
fi

# Number of instances to create
N=$1

# Create a temporary file to store the filtered data
temp_file=$(mktemp)

# Read the input, filter for CUDA version >= 12.2, and save to temporary file
while IFS= read -r line; do
    if [[ $line =~ ^[0-9]+ ]]; then
        cuda_version=$(echo $line | awk '{print $2}')
        if (( $(echo "$cuda_version >= 12.2" | bc -l) )); then
            echo $line >> "$temp_file"
        fi
    fi
done

# Generate the vastai create instance commands
echo "#!/bin/bash"
echo ""
echo "# Commands to create $N vast.ai instances"

awk -v n=$N 'NR <= n {print "vastai create instance " $1 " --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel --disk 30"}' "$temp_file"

# Clean up the temporary file
rm "$temp_file"
