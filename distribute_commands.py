import subprocess
import itertools
import logging
import argparse
import queue
import threading
import os
import time

def rsync_from_vm(vm_name, remote_path, local_path):
    """Sync files from VM to local machine"""
    os.makedirs(local_path, exist_ok=True)
    
    rsync_command = [
        'rsync',
        '-avz',  # archive mode, verbose, compress
        f'{vm_name}:{remote_path}',
        local_path
    ]
    
    try:
        result = subprocess.run(rsync_command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Rsync from VM failed with error: {e}")
        logging.error(e.stderr)
        return False

def destroy_vm(vm_name):
    """Destroy the VM using the destroy-vm script"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'destroy-vm')
        result = subprocess.run([script_path, vm_name], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to destroy VM {vm_name}: {e}")
        logging.error(e.stderr)
        return False

def rsync_version_files(source_dir, remote_host, remote_path, version):
    # Find all files in source_dir that have 'version' in their name
    files_to_sync = [
        os.path.join(root, file)
        for root, _, files in os.walk(source_dir)
        for file in files
        if version in file
    ]

    # Construct the rsync command
    rsync_command = [
        'rsync',
        '-avz',  # archive mode, verbose, compress
    ] + files_to_sync + [
        f'{remote_host}:{remote_path}'
    ]

    # Execute the rsync command
    try:
        result = subprocess.run(rsync_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Rsync failed with error: {e}")
        print(e.stderr)

def run_command_on_vm(vm_name, command):
    try:
        format_aux = ' '.join(command).split('--version ')[-1]
        version = format_aux.split(' ')[0]
        rsync_version_files('model_bin', vm_name, '/workspace/alpr/model_bin/', version+'.pth')
    except BaseException as e:
        logging.error(f'Error on version rsync: {command}', exc_info=True)

    ssh_command = f"ssh {vm_name} 'cd /workspace/alpr && {command}'"
    print(f"Running on {vm_name}: {command}")
    try:
        subprocess.run(ssh_command, shell=True, check=True)

        # After successful command execution, sync back the files
        print(f"Syncing files back from {vm_name}")
        model_sync_success = rsync_from_vm(
            vm_name, 
            '/workspace/alpr/model_bin/', 
            'model_bin/'
        )
        config_sync_success = rsync_from_vm(
            vm_name, 
            '/workspace/alpr/configs/', 
            'configs/'
        )

        # If both syncs were successful, destroy the VM
        if model_sync_success and config_sync_success:
            print(f"Destroying VM {vm_name}")
            destroy_success = destroy_vm(vm_name)
            if not destroy_success:
                logging.error(f"Failed to destroy VM {vm_name}")

        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running command on {vm_name}: {e}")
        return False

def vm_worker(vm_name, command_queue):
    while True:
        try:
            command = command_queue.get(block=True, timeout=1)  # Wait for 1 second for a new command
            success = run_command_on_vm(vm_name, command)
            if not success:
                # If command fails, put it back in the queue
                command_queue.put(command)
            command_queue.task_done()
        except queue.Empty:
            # No more commands in the queue
            break

def main(N, initial_version):
    # Define the argument combinations you want to try
    configurations = {
        'patch_size': ['48 8'],
        'img_height': ['48'],
        'img_width': ['192'],
        'batch_size': ['1536'],
        'epochs': ['400'],
        'embed_dim': ['624'],
        'num_heads': ['12'],
        'num_encoder_layers': ['12', '8'],
        'num_decoder_layers': ['4', '3'],
        'max_sequence_length': ['7'],
        'dropout': ['0.4'],
        'emb_dropout': ['0.4'],
        'norm_image': ['1'],
        'overlap': ['0'],
        'start_lr': [str(1e-3)],
        'plateau_thr': ['500'],
        'wandb': ['']
    }

    # Generate all combinations of arguments
    keys, values = zip(*configurations.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Prepare command queue
    command_queue = queue.Queue()
    for idx, combo in enumerate(combinations):
        args = ' '.join([f'--{key} {value}' for key, value in combo.items()])
        command = f"/opt/conda/bin/python maskocr_train.py {args} --version v{idx+1+initial_version}"
        command_queue.put(command)

    # Create and start a thread for each VM
    threads = []
    for i in range(N):
        vm_name = f'cuda-dev-{i+1}'
        thread = threading.Thread(target=vm_worker, args=(vm_name, command_queue))
        thread.start()
        threads.append(thread)

    # Wait for all commands to be processed
    command_queue.join()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyTorch training on Vast.ai VMs")
    parser.add_argument("--num_vms", type=int, help="Number of VMs")
    parser.add_argument("--initial_version", type=int, default=0, help="Version start")
    args = parser.parse_args()
    main(args.num_vms, args.initial_version)