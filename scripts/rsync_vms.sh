#!/bin/bash

# Calculate the number of VMs
actual_vms=$(vastai show instances | wc -l)
num_vms=$((actual_vms - 1))

# Check if we have any VMs
if [ "$num_vms" -eq 0 ]; then
    echo "Error: No VMs detected. Please ensure you have running instances."
    exit 1
fi

echo "Detected $num_vms VMs."

# Function to sync files from a single VM
sync_vm() {
    local vm_num=$1
    echo "Syncing from cuda-dev-$vm_num..."
    rsync -avr "cuda-dev-$vm_num:/workspace/alpr/model_bin/" "model_bin/"
    echo "Sync complete for cuda-dev-$vm_num."
}

# Export the function so it's available to parallel
export -f sync_vm

# Sync files from each VM in parallel
echo "Syncing files from VMs..."
seq 1 $((num_vms - 1)) | parallel -j0 sync_vm

echo "All syncs complete."
