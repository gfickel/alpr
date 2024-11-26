import json
import logging
import os
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional
import itertools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VastAIManager:
    def __init__(self, config_path: str = "configs/training_configs.json"):
        """Initialize VastAI manager with configuration file."""
        self.config = self._load_config(config_path)
        self.base_dir = Path(os.getcwd())
        self._cmd_lock = threading.Lock()
        self._ssh_lock = threading.Lock()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _calculate_num_vms(self, network: str) -> int:
        """Calculate number of VMs needed based on configuration combinations."""
        network_config = self.config['networks'][network]
        configs = network_config['configurations']
        
        # Generate all possible combinations of parameters
        keys = configs.keys()
        values = [configs[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        return len(combinations)
    
    def _run_command(self, command: str, shell: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and handle errors."""
        try:
            result = subprocess.run(command, shell=shell, check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error output: {e.stderr}")
            raise

    def _generate_vast_search_query(self) -> str:
        """Generate VastAI search query from config."""
        params = self.config['vm_config']['search_params']
        query = (f"reliability>{params['reliability']} "
                f"inet_up>{params['min_inet_up']} "
                f"inet_down>{params['min_inet_down']} "
                f"cpu_cores>{params['min_cpu_cores']}")
        return query

    def _wait_for_vms(self, num_vms: int, max_attempts: int = 300) -> bool:
        """Wait for VMs to be in running state."""
        attempt = 0
        while attempt < max_attempts:
            output = self._run_command("vastai show instances")
            running_vms = 0
            for line in output.stdout.splitlines()[1:]:  # Skip header
                if "running" in line:
                    running_vms += 1
            
            if running_vms >= num_vms:
                logger.info(f"All {num_vms} VMs are running")
                # Add extra wait to ensure VMs are fully initialized
                time.sleep(10)
                return True
            
            logger.info(f"Waiting for VMs to start ({running_vms}/{num_vms} running)")
            time.sleep(1)
            attempt += 1
        
        return False

    def _setup_ssh_config(self):
        """Update SSH config for VastAI instances."""
        output = self._run_command("vastai show instances")
        self._run_command(f"echo '{output.stdout}' | ./scripts/update_ssh_config.sh")

    def _generate_vm_commands(self, network: str, start_version: int) -> Dict[int, str]:
        """Generate commands for each VM based on configuration combinations."""
        network_config = self.config['networks'][network]
        configs = network_config['configurations']
        
        # Generate all possible combinations of parameters
        keys = configs.keys()
        values = [configs[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        vm_commands = {}
        vm_idx = 0
        
        for idx, combo in enumerate(combinations):
            args = ' '.join([f'--{key} {value}' for key, value in zip(keys, combo)])
            version = f'v{idx+1+start_version}'
            
            if network == 'detection':
                    command = (f"/opt/conda/bin/python {network_config['script']} "
                             f"{args} --version {version} "
                             f"--dataset_path ../")
                    vm_commands[vm_idx] = command
                    vm_idx += 1
            else:
                command = f"/opt/conda/bin/python {network_config['script']} {args} --version {version} --dataset_path ../"
                vm_commands[vm_idx] = command
                vm_idx += 1
        
        return vm_commands

    def _check_ssh_connection(self, vm_name: str, max_retries: int = 30) -> bool:
        """Check if SSH connection to VM is available."""
        for i in range(max_retries):
            try:
                result = self._run_command(f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no {vm_name} 'echo test'")
                if result.returncode == 0:
                    logger.info(f"SSH connection established to {vm_name}")
                    return True
            except subprocess.CalledProcessError:
                if i == max_retries - 1:
                    logger.error(f"Failed to establish SSH connection to {vm_name} after {max_retries} attempts")
                    return False
                logger.info(f"Waiting for SSH connection to {vm_name} (attempt {i+1}/{max_retries})")
                time.sleep(5)  # Increased wait time between retries
        
        return False

    def _setup_vm(self, vm_name: str, network: str) -> bool:
        """Setup a single VM with required files and datasets."""
        try:
            logger.info(f"Setting up {vm_name}")
            
            if not self._check_ssh_connection(vm_name):
                logger.error(f"Could not establish SSH connection to {vm_name}")
                return False

            with self._ssh_lock:
                self._run_command(f"rsync -avz -e 'ssh -o StrictHostKeyChecking=no' ./ {vm_name}:/workspace/alpr/")
                
                if os.path.exists(".netrc"):
                    self._run_command(f"scp -o StrictHostKeyChecking=no .netrc {vm_name}:/root/")

            network_config = self.config['networks'][network]
            setup_commands = [
                "source /root/.bashrc",
                "apt install -y unzip cmake build-essential g++",
                "cd /workspace/alpr/ && /opt/conda/bin/python -m pip install -r requirements.txt"
            ]
            
            for dataset in network_config['datasets']:
                setup_commands.extend([
                    f"/opt/conda/bin/python /workspace/alpr/scripts/download_files.py {dataset['id']} {dataset['extract_dir']}",
                    f"unzip -q /workspace/{dataset['filename']} -d {dataset['extract_dir']}"
                ])
            
            setup_script = " && ".join(setup_commands)
            self._run_command(f"ssh -o StrictHostKeyChecking=no {vm_name} '{setup_script}'")
            
            logger.info(f"Setup completed for {vm_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up {vm_name}: {str(e)}")
            return False


    def _create_coordinator_vm(self) -> bool:
        """Create the coordinator VM (cuda-dev-0) with minimal requirements."""
        logger.info("Creating coordinator VM...")
        # Using the same search criteria as in the original script
        search_query = "reliability>0.98 dph<0.1 inet_up>100 inet_down>500"
        
        # Create coordinator VM
        search_result = self._run_command(f"vastai search offers '{search_query}' -o dph")
        self._run_command(f"echo '{search_result.stdout}' | ./scripts/create_vms.sh 1 > scripts/vastai_commands.sh")
        
        # Execute the commands
        self._run_command("chmod +x scripts/vastai_commands.sh")
        self._run_command("./scripts/vastai_commands.sh")
        
        # Wait for coordinator VM to start
        if not self._wait_for_vms(1):
            logger.error("Timeout waiting for coordinator VM to start")
            return False
            
        # Setup SSH config for coordinator VM
        self._setup_ssh_config()
        return True

    def _create_training_vms(self, num_vms: int, gpu_name: str) -> bool:
        """Create the training VMs with specified GPU requirements."""
        logger.info(f"Creating {num_vms} training VMs...")
        search_query = f"{self._generate_vast_search_query()} gpu_name={gpu_name}"
        
        # Create training VMs
        search_result = self._run_command(f"vastai search offers '{search_query}' -o dph")
        self._run_command(f"echo '{search_result.stdout}' | ./scripts/create_vms.sh {num_vms} > scripts/vastai_commands.sh")
        
        # Execute the commands
        self._run_command("chmod +x scripts/vastai_commands.sh")
        self._run_command("./scripts/vastai_commands.sh")
        
        # Wait for training VMs to start
        if not self._wait_for_vms(num_vms + 1):  # +1 for coordinator VM
            logger.error("Timeout waiting for training VMs to start")
            return False
        
        # Update SSH config for all VMs
        self._setup_ssh_config()
        return True

    def _setup_coordinator_vm(self) -> bool:
        """Setup coordinator VM with minimal requirements."""
        try:
            vm_name = "cuda-dev-0"
            logger.info(f"Setting up coordinator VM {vm_name}")
            
            if not self._check_ssh_connection(vm_name):
                logger.error(f"Could not establish SSH connection to {vm_name}")
                return False

            # Basic setup without datasets
            with self._ssh_lock:
                self._run_command(f"rsync -avz -e 'ssh -o StrictHostKeyChecking=no' ./ {vm_name}:/workspace/alpr/")
                
                if os.path.exists(".netrc"):
                    self._run_command(f"scp -o StrictHostKeyChecking=no .netrc {vm_name}:/root/")

            # Only install basic requirements
            setup_commands = [
                "source /root/.bashrc",
                "apt install -y parallel",
                "cd /workspace/alpr/ && /opt/conda/bin/python -m pip install vastai"
            ]
            
            setup_script = " && ".join(setup_commands)
            self._run_command(f"ssh -o StrictHostKeyChecking=no {vm_name} '{setup_script}'")
            
            logger.info(f"Setup completed for coordinator VM {vm_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up coordinator VM {vm_name}: {str(e)}")
            return False

    def _setup_training_vm(self, vm_name: str, network: str) -> bool:
        """Setup a single training VM with required files and datasets."""
        try:
            logger.info(f"Setting up training VM {vm_name}")
            
            # Keep trying to establish SSH connection
            max_retries = 30
            connected = False
            for i in range(max_retries):
                try:
                    # Use SSH lock only for initial connection when known_hosts might be modified
                    with self._ssh_lock:
                        result = self._run_command(
                            f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no {vm_name} 'echo test'"
                        )
                    if result.returncode == 0:
                        connected = True
                        break
                except subprocess.CalledProcessError:
                    if i < max_retries - 1:
                        time.sleep(20)

            if not connected:
                logger.error(f"Could not establish SSH connection to {vm_name}")
                return False

            logger.info(f"Starting setup for {vm_name}")
            
            # Regular file transfers can happen in parallel
            logger.info(f"Transferring files to {vm_name}")
            self._run_command(f"rsync -avz -e 'ssh -o StrictHostKeyChecking=no' ./ {vm_name}:/workspace/alpr/")
            
            if os.path.exists(".netrc"):
                self._run_command(f"scp -o StrictHostKeyChecking=no .netrc {vm_name}:/root/")

            network_config = self.config['networks'][network]
            setup_commands = [
                "source /root/.bashrc",
                "apt install -y unzip cmake build-essential g++",
                "cd /workspace/alpr/ && /opt/conda/bin/python -m pip install -r requirements.txt"
            ]
            
            for dataset in network_config['datasets']:
                setup_commands.extend([
                    f"/opt/conda/bin/python /workspace/alpr/scripts/download_files.py {dataset['id']} {dataset['extract_dir']}",
                    f"unzip -q /workspace/{dataset['filename']} -d {dataset['extract_dir']}"
                ])
            
            setup_script = " && ".join(setup_commands)
            logger.info(f"Running setup commands on {vm_name}")
            self._run_command(f"ssh -o StrictHostKeyChecking=no {vm_name} '{setup_script}'")
            
            logger.info(f"Setup completed for training VM {vm_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up training VM {vm_name}: {str(e)}")
            return False

    def _setup_vms_parallel(self, num_vms: int, network: str) -> bool:
        """Setup multiple training VMs in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=num_vms) as executor:
            # Submit all VM setups
            future_to_vm = {
                executor.submit(self._setup_training_vm, f"cuda-dev-{i+1}", network): i 
                for i in range(num_vms)
            }
            
            # Wait for all setups to complete
            failed_vms = []
            for future in as_completed(future_to_vm):
                vm_idx = future_to_vm[future]
                try:
                    success = future.result()
                    if not success:
                        failed_vms.append(vm_idx + 1)
                except Exception as e:
                    logger.error(f"VM cuda-dev-{vm_idx + 1} setup failed with error: {str(e)}")
                    failed_vms.append(vm_idx + 1)
            
            if failed_vms:
                logger.error(f"Setup failed for VMs: {failed_vms}")
                return False
                
            return True

    def _setup_coordinator_script(self, vm_commands: Dict[int, str]) -> bool:
        """Setup and execute the coordinator script with proper escaping."""
        try:
            # First, create the base script content
            coordinator_script = [
                "#!/bin/bash",
                "cd /workspace/alpr",
                "sleep 30",  # Wait for VMs to be ready
            ]
            
            # Create individual command files for each VM
            for vm_idx, command in vm_commands.items():
                vm_name = f"cuda-dev-{vm_idx + 1}"
                cmd_file = f"cmd_{vm_idx}.sh"
                
                # Create command file on coordinator VM
                self._run_command(
                    f'''ssh cuda-dev-0 "echo '#!/bin/bash
    cd /workspace/alpr
    {command}' > /workspace/alpr/{cmd_file} && chmod +x /workspace/alpr/{cmd_file}"'''
                )
                
                # Add execution of this command file to coordinator script
                coordinator_script.append(f'ssh -o StrictHostKeyChecking=no {vm_name} "/workspace/alpr/{cmd_file}" &')
            
            coordinator_script.append("wait")  # Wait for all processes
            
            # Write the main coordinator script
            script_content = "\n".join(coordinator_script)
            self._run_command(
                f'''ssh cuda-dev-0 "echo '{script_content}' > /workspace/alpr/start_training.sh && chmod +x /workspace/alpr/start_training.sh"'''
            )
            
            # Execute the coordinator script in a detached screen session
            self._run_command(
                'ssh cuda-dev-0 "apt-get update && apt-get install -y screen && ' + 
                'screen -dmS training /workspace/alpr/start_training.sh"'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up coordinator script: {str(e)}")
            return False

    def setup_training(self, network: str, gpu_name: Optional[str] = None, 
                      start_version: int = 0) -> bool:
        """
        Setup and start training on VastAI VMs.
        """
        if network not in self.config['networks']:
            raise ValueError(f"Network '{network}' not found in config")
        
        # Calculate required number of training VMs based on configurations
        num_training_vms = self._calculate_num_vms(network)
        logger.info(f"Calculated {num_training_vms} VMs needed for training")
        
        # Create coordinator VM first with minimal requirements
        if not self._create_coordinator_vm():
            logger.error("Failed to create coordinator VM")
            return False
            
        # Setup coordinator VM (without datasets)
        logger.info("Setting up coordinator VM...")
        if not self._setup_coordinator_vm():
            logger.error("Coordinator VM setup failed")
            return False

        # Create training VMs with GPU requirements
        gpu = gpu_name or self.config['vm_config']['default_gpu']
        if not self._create_training_vms(num_training_vms, gpu):
            logger.error("Failed to create training VMs")
            return False
        
        # Setup training VMs in parallel
        logger.info("Setting up training VMs in parallel...")
        if not self._setup_vms_parallel(num_training_vms, network):
            logger.error("Training VMs setup failed")
            return False

        # Generate commands for each VM
        vm_commands = self._generate_vm_commands(network, start_version)
        
        try:
            # Ensure SSH directory exists on coordinator VM
            self._run_command('ssh cuda-dev-0 "mkdir -p ~/.ssh && chmod 700 ~/.ssh"')
            
            # Copy all SSH configuration files to coordinator VM
            ssh_path = os.path.expanduser("~/.ssh")
            for ssh_file in ['config', 'id_rsa', 'id_rsa.pub', 'known_hosts']:
                if os.path.exists(os.path.join(ssh_path, ssh_file)):
                    self._run_command(f'scp {ssh_path}/{ssh_file} cuda-dev-0:~/.ssh/')
            
            # Set proper permissions on coordinator VM
            self._run_command('ssh cuda-dev-0 "chmod 600 ~/.ssh/id_rsa ~/.ssh/id_rsa.pub ~/.ssh/config ~/.ssh/known_hosts"')
            
            # Create the training script locally
            with open('start_training.sh', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('cd /workspace/alpr\n\n')
                
                # Add each training command
                for vm_idx, command in vm_commands.items():
                    vm_name = f"cuda-dev-{vm_idx + 1}"
                    f.write(f'ssh -o StrictHostKeyChecking=no {vm_name} "{command}" &\n')
                
                f.write('\nwait\n')  # Wait for all processes to complete
            
            # Make the script executable
            os.chmod('start_training.sh', 0o755)
            
            # Copy the script to coordinator VM
            self._run_command('scp start_training.sh cuda-dev-0:/workspace/alpr/')
            
            # Run the script on coordinator VM
            self._run_command('ssh cuda-dev-0 "cd /workspace/alpr && nohup ./start_training.sh > training.log 2>&1 &"')
            
            # Clean up local script
            os.remove('start_training.sh')
            
            logger.info(f"Training started on all {len(vm_commands)} VMs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup and manage VastAI training")
    parser.add_argument("--network", type=str, required=True, help="Network to train (from config)")
    parser.add_argument("--gpu", type=str, help="GPU type to request")
    parser.add_argument("--start_version", type=int, default=0, help="Starting version number")
    
    args = parser.parse_args()
    
    manager = VastAIManager()
    manager.setup_training(args.network, args.gpu, args.start_version)