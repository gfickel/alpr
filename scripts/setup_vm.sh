#!/bin/bash

# Function to add host keys automatically
add_host_keys() {
    local vm_name=$1
    local ssh_config=$(grep "^host $vm_name$" -A 3 ~/.ssh/config)
    local hostname=$(echo "$ssh_config" | grep "HostName" | awk '{print $2}')
    local port=$(echo "$ssh_config" | grep "Port" | awk '{print $2}')
    if [[ -n $hostname && -n $port ]]; then
        echo "Adding host key for $vm_name ($hostname:$port)"
        ssh-keyscan -H -p $port $hostname >> ~/.ssh/known_hosts 2>/dev/null
    else
        echo "Couldn't find hostname or port for $vm_name in ~/.ssh/config"
    fi
}

# Function to check if all VMs are running
check_vm_status() {
    local num_vms=$1
    local all_running=false
    local max_attempts=600  # Maximum number of attempts
    local attempt=0

    while [ "$all_running" = false ] && [ $attempt -lt $max_attempts ]; do
        all_running=true
        output=$(vastai show instances)
        echo "$output"  # Print the output for debugging
        for i in $(seq 1 $((num_vms))); do
            # Extract the status based on the instance ID
            instance_id=$(echo "$output" | awk 'NR>1 {print $1}' | sed -n "$((i+1))p")
            if [ -n "$instance_id" ]; then
                status=$(echo "$output" | awk -v id="$instance_id" '$1 == id {print $3}')
                echo "VM cuda-dev-$i (ID: $instance_id) status: $status"  # Debug output
                if [ "$status" != "running" ]; then
                    all_running=false
                    echo "VM cuda-dev-$i (ID: $instance_id) is not running yet (status: $status)"
                    break
                fi
            else
                echo "Could not find instance ID for VM cuda-dev-$i"
                all_running=false
                break
            fi
        done

        if [ "$all_running" = false ]; then
            echo "Waiting for all VMs to be in 'running' state..."
            sleep 1
            ((attempt++))
        fi
    done

    if [ $attempt -eq $max_attempts ]; then
        echo "Timeout: Not all VMs are in 'running' state after 5 minutes."
        exit 1
    fi

    echo "All VMs are now in 'running' state."
    echo "Waiting an additional 10 seconds for VMs to fully initialize..."
    sleep 10
}

# Function to setup a single VM
setup_vm() {
    local vm_name=$1
    echo "Setting up $vm_name..."
    # SSH options
    local ssh_options="-o BatchMode=yes -o StrictHostKeyChecking=no"
    ssh $ssh_options $vm_name "touch ~/.no_auto_tmux"
    
    # Assuming 'alpr' directory is in the same directory as this script
    echo "Syncing files to $vm_name"
    rsync -avz -e "ssh $ssh_options" "../alpr/" "$vm_name:/workspace/alpr/"
    
    # Assuming 'wandb/.netrc' is in the same directory as this script
    scp $ssh_options ".netrc" "$vm_name:/root/"
    
    # Perform remaining setup tasks
    ssh $ssh_options $vm_name << EOF
        # Source environment setup files
        source /root/.bashrc
        apt install -y unzip
        cd /workspace/alpr/ && /opt/conda/bin/python -m pip install -r requirements.txt
        /opt/conda/bin/python /workspace/alpr/download_files.py 1-j2ua0SBlRBmdK0AP47GV9AxOXKAWP7d /workspace/
        unzip -q /workspace/output_images_plates_gt.zip -d /workspace/
        /opt/conda/bin/python /workspace/alpr/download_files.py 1b7bY2JP_XcQ2u4XNNWt9b5q03R0YJjDO /workspace/
        unzip -q /workspace/plates_ccpd_weather.zip -d /workspace/
EOF
    echo "Setup completed for $vm_name"
}

export -f setup_vm
export -f add_host_keys

# Main script
if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_vms>"
    exit 1
fi

num_vms=$1

# Check VM status before proceeding
check_vm_status $num_vms

# Add host keys for all VMs
for i in $(seq 1 $((num_vms))); do
    add_host_keys "cuda-dev-$i"
done

# Run setup_vm in parallel for all VMs
seq 1 $((num_vms)) | parallel -j$num_vms setup_vm "cuda-dev-{}"

echo "All VMs have been set up."