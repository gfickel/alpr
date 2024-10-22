#!/bin/bash

# Function to update or add an SSH config entry
update_ssh_config() {
    local host="$1"
    local hostname="$2"
    local port="$3"
    local config_file="$HOME/.ssh/config"
    local temp_file=$(mktemp)

    # Create new content for this host
    local new_content="host $host
    HostName $hostname
    User root
    Port $port
"

    # Check if the host already exists in the config
    if grep -q "^host $host$" "$config_file"; then
        # Replace existing entry
        awk -v host="$host" -v new="$new_content" '
        $0 ~ "^host "host"$" {
            print new
            found = 1
            next
        }
        found && /^host / {
            found = 0
        }
        !found
        ' "$config_file" > "$temp_file"
    else
        # Add new entry
        cp "$config_file" "$temp_file"
        echo "" >> "$temp_file"
        echo "$new_content" >> "$temp_file"
    fi

    # Replace the original file with the updated one
    mv "$temp_file" "$config_file"
}

# Main script
instance_count=0

while IFS= read -r line; do
    if [[ $line =~ ^[0-9]+ ]]; then
        id=$(echo "$line" | awk '{print $1}')
        hostname=$(echo "$line" | awk '{print $10}')
        port=$(echo "$line" | awk '{print $11}')
        
        if [[ $hostname != "-" && $port != "-" ]]; then
            host="cuda-dev-$instance_count"
            update_ssh_config "$host" "$hostname" "$port"
            echo "Updated SSH config for $host (ID: $id, HostName: $hostname, Port: $port)"
            ((instance_count++))
        fi
    fi
done

echo "SSH config updated for $instance_count instances."
