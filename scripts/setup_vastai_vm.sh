#!/bin/bash

# Function to check if all VMs are running
check_vm_status() {
    local num_vms=$1
    local all_running=false
    local max_attempts=300  # Maximum number of attempts (5 minutes with 10-second intervals)
    local attempt=0

    while [ "$all_running" = false ] && [ $attempt -lt $max_attempts ]; do
        all_running=true
        output=$(vastai show instances)
        echo "$output"  # Print the output for debugging
        for i in $(seq 0 $((num_vms - 1))); do
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

# Function to add host keys automatically
add_host_keys() {
    local vm_name=$1
    local ssh_config=$(grep "^host $vm_name$" -A 3 ~/.ssh/config)
    local hostname=$(echo "$ssh_config" | grep "HostName" | awk '{print $2}')
    local port=$(echo "$ssh_config" | grep "Port" | awk '{print $2}')
    if [[ -n $hostname && -n $port ]]; then
        echo "Adding host key for $vm_name ($hostname:$port)"
        ssh-keyscan -T 10 -H -p $port $hostname >> ~/.ssh/known_hosts 2>/dev/null
    else
        echo "Couldn't find hostname or port for $vm_name in ~/.ssh/config"
    fi
}

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

# Create the initial VM
echo "Creating initial VM..."
vastai search offers "reliability>0.98 dph<0.1 inet_up>100 inet_down>100" -o dph | ./scripts/create_vms.sh 1 > scripts/vastai_commands.sh
bash scripts/vastai_commands.sh

# Check if the VM was created successfully
actual_vms=$(vastai show instances | wc -l)
actual_vms=$((actual_vms - 1))  # Subtract 1 to account for header

if [ "$actual_vms" -eq 0 ]; then
    echo "Error: No VM was successfully created. Exiting."
    exit 1
fi

echo "Initial VM created successfully."

# Update SSH config
echo "Updating SSH config..."
vastai show instances | ./scripts/update_ssh_config.sh

# Check VM status
echo "Checking VM status..."
check_vm_status 1

# Add host keys to known_hosts
echo "Adding host keys..."
add_host_keys cuda-dev-0

# Install vastai on the VM
echo "Installing vastai on the VM..."
ssh -o StrictHostKeyChecking=no cuda-dev-0 "/opt/conda/bin/python -m pip install vastai"

# Install vastai and bc on the VM
echo "Installing vastai and bc on the VM..."
ssh -o StrictHostKeyChecking=no cuda-dev-0 "
    /opt/conda/bin/python -m pip install vastai
    sudo apt-get update
    sudo apt-get install -y bc parallel
"

# Generate start_vms_train.sh script
echo "Generating start_vms_train.sh script..."
cat << EOF > scripts/start_vms_train.sh
#!/bin/bash

scripts/cloud_setup.sh $requested_vms $gpu_name $network_type $initial_version
EOF

# Copy the Vast.ai API key
echo "Copying Vast.ai API key..."
scp ~/.vast_api_key cuda-dev-0:~/.vast_api_key
scp ~/.ssh/id_* cuda-dev-0:~/.ssh/

# Copy the current directory to the VM
echo "Copying current directory to VM..."
rsync -avz --exclude '.git' --exclude 'venv' --exclude '__pycache__' ./ cuda-dev-0:/workspace/alpr/

echo "Setup complete. The cloud_setup.sh script is now running on the initial VM."
echo "You can safely close this terminal or shut down your local machine."
echo "To check on the progress later, you can SSH into the VM using:"
echo "ssh cuda-dev-0"

chmod +x scripts/start_vms_train.sh
echo "Generated start_vms_train.sh with command: cloud_setup.sh $requested_vms $gpu_name $network_type $initial_version"

ssh cuda-dev-0 "sudo sh -c 'echo \"PATH=\\\"/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin\\\"\" > /etc/environment'"
ssh cuda-dev-0 "bash -l -c 'cd /workspace/alpr && bash scripts/cloud_setup.sh $requested_vms $gpu_name $network_type $initial_version' > /workspace/alpr/scripts/cloud_setup.log 2>&1 &"
sleep 3
ssh cuda-dev-0 "tail -f /workspace/alpr/scripts/cloud_setup.log"