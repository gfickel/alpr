#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <number_of_vms> <gpu_name> [initial_version]"
    echo "Example: $0 2 RTX_3060 5"
    echo "If initial_version is not provided, it defaults to 0"
    exit 1
fi

# Assign command-line arguments to variables
requested_vms=$1
gpu_name=$2
initial_version=${3:-0}  # Use the third argument if provided, otherwise default to 0

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

# Update num_vms to the actual number of VMs created
num_vms=$actual_vms

# Continue with the rest of the script using the updated num_vms
vastai show instances | ./scripts/update_ssh_config.sh
./scripts/setup_vm.sh "$num_vms"
python distribute_commands.py --num_vms "$num_vms" --initial_version "$initial_version"

# New section: Sync files from each VM
echo "Syncing files from VMs..."
for i in $(seq 1 $((num_vms))); do
    echo "Syncing from cuda-dev-$i..."
    rsync -avr "cuda-dev-$i:/workspace/alpr/" "."
done

echo "Sync completed for all VMs."

./scripts/destroy_vastai_instances.sh --keep-first