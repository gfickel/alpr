#!/bin/bash

# Check the actual number of VMs created
# We subtract 1 from the line count to account for the header line and one for the first Main VM
actual_vms=$(vastai show instances | wc -l)
actual_vms=$((actual_vms - 1 - 1))

echo "Number of VMs detected: $actual_vms"

# Command to run on each VM
command="python upload_files.py model_bin/* configs/*"

# Loop through each VM and run the command
for ((i=1; i<=actual_vms; i++)); do
    vm_name="cuda-dev-$i"
    output_file="uploaded_files_VM_$i.txt"
    
    echo "Running command on $vm_name"
    
    # Use ssh to run the command on the VM and save the output locally
    ssh "$vm_name" "$command" > "$output_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Command executed successfully on $vm_name. Output saved to $output_file"
    else
        echo "Error executing command on $vm_name. Check $output_file for details"
    fi
done

echo "Script execution completed."