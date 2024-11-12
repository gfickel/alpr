#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <number_of_vms> <gpu_name> <network_type> [initial_version]"
    echo "Example: $0 2 RTX_3060 detection 5"
    echo "network_type can be 'detection' or 'maskocr'"
    echo "If initial_version is not provided, it defaults to 0"
    exit 1
fi

# Assign command-line arguments to variables
requested_vms=$1
gpu_name=$2
network_type=$3
initial_version=${4:-0}  # Use the fourth argument if provided, otherwise default to 0

# Validate network_type
if [ "$network_type" != "detection" ] && [ "$network_type" != "maskocr" ]; then
    echo "Error: network_type must be either 'detection' or 'maskocr'"
    exit 1
fi

# Use the variables in the commands
vastai search offers "reliability>0.98 num_gpus=1 inet_up>100 inet_down>100 cpu_cores>4 gpu_name=${gpu_name}" -o dph | ./scripts/create_vms.sh "$requested_vms" > scripts/vastai_commands.sh
bash scripts/vastai_commands.sh

# Check the actual number of VMs created
# We subtract 1 from the line count to account for the header line and one for the first Main VM
actual_vms=$(vastai show instances | wc -l)
actual_vms=$((actual_vms - 1 - 1))

if [ "$actual_vms" -eq 0 ]; then
    echo "Error: No VMs were successfully created. Exiting."
    exit 1
fi

if [ "$actual_vms" -lt "$requested_vms" ]; then
    echo "Warning: Only $actual_vms out of $requested_vms requested VMs were successfully created."
else
    echo "All $requested_vms VMs were successfully created."
fi

# Continue with the rest of the script using the actual number of VMs created
num_vms=$actual_vms

# Update SSH config and setup VMs with the network type parameter
vastai show instances | ./scripts/update_ssh_config.sh
./scripts/setup_vm.sh "$num_vms" "$network_type"

# Run distribute_commands.py with the network type parameter
python distribute_commands.py --num_vms "$num_vms" --initial_version "$initial_version" --network "$network_type"

# New section: Sync files from each VM
echo "Syncing files from VMs..."
for i in $(seq 1 $((num_vms))); do
    echo "Syncing from cuda-dev-$i..."
    rsync -avr "cuda-dev-$i:/workspace/alpr/" "."
done

echo "Sync completed for all VMs."

./scripts/destroy_vastai_instances.sh --keep-first