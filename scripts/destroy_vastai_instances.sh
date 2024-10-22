#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [--keep-first]"
    echo "  --keep-first    Keep the first instance running (optional)"
    exit 1
}

# Function to destroy an instance with retry mechanism
destroy_instance() {
    local id=$1
    local max_retries=60
    local retry_delay=5

    for ((i=1; i<=max_retries; i++)); do
        echo "Attempting to destroy instance $id (Try $i of $max_retries)..."
        output=$(vastai destroy instance $id 2>&1)
        
        if [[ $output == *"failed"* ]]; then
            echo "Received an error. Retrying in $retry_delay seconds..."
            sleep $retry_delay
        else
            echo "$output"
            return 0
        fi
    done

    echo "Failed to destroy instance $id after $max_retries attempts."
    return 1
}

# Parse command line arguments
keep_first=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --keep-first) keep_first=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Get the output of vastai show instances
output=$(vastai show instances)

# Check if the command was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to get instance list. Make sure 'vastai' is installed and configured correctly."
    exit 1
fi

# Parse the output and extract instance IDs
instance_ids=$(echo "$output" | awk 'NR>1 {print $1}')

# Check if any instances were found
if [ -z "$instance_ids" ]; then
    echo "No instances found."
    exit 0
fi

# Log the instances that will be destroyed
echo "The following instances will be destroyed:"
if [ "$keep_first" = true ]; then
    echo "$instance_ids" | sed '1d'
    echo "The first instance will be kept running."
else
    echo "$instance_ids"
fi

# Destroy each instance
first=true
echo "$instance_ids" | while read id; do
    if [ "$keep_first" = true ] && [ "$first" = true ]; then
        echo "Keeping instance $id running."
        first=false
    else
        destroy_instance $id
    fi
done

echo "Operation completed."