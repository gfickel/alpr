#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <vm-name>"
    echo "Example: $0 cuda-dev-1"
    exit 1
}

# Check if VM name is provided
if [ $# -ne 1 ]; then
    usage
fi

vm_name=$1

# Get instance ID from vastai show instances based on hostname in SSH config
ssh_config="$HOME/.ssh/config"
hostname=$(awk -v host="$vm_name" '
    $1 == "host" && $2 == host {in_block=1; next}
    in_block && $1 == "HostName" {print $2; exit}
    in_block && $1 ~ /^host/ {exit}
' "$ssh_config")

if [ -z "$hostname" ]; then
    echo "Error: Could not find hostname for VM $vm_name in SSH config"
    exit 1
fi

# Get instance ID from vastai show instances
instance_id=$(vastai show instances | awk -v hostname="$hostname" '$10 == hostname {print $1; exit}')

if [ -z "$instance_id" ]; then
    echo "Error: Could not find instance ID for hostname $hostname"
    exit 1
fi

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

# Destroy the instance
destroy_instance "$instance_id"