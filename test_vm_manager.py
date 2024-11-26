import json

class VastAIManager:
    def __init__(self, config_path: str = "training_config.json"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return json.load(f)
        
    def _generate_vm_commands(self, network: str, start_version: int) -> dict:
        """Generate commands for each VM based on configuration combinations."""
        network_config = self.config['networks'][network]
        configs = network_config['configurations']
        
        import itertools
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
                            f"--dataset_path ..")
                vm_commands[vm_idx] = command
                vm_idx += 1
            else:
                command = f"/opt/conda/bin/python {network_config['script']} {args} --version {version}"
                vm_commands[vm_idx] = command
                vm_idx += 1
        
        return vm_commands

# Test the function
manager = VastAIManager('configs/training_configs.json')

# Test detection network
print("\nDetection network commands:")
detection_commands = manager._generate_vm_commands('detection', 0)
for vm_idx, command in detection_commands.items():
    print(f"VM {vm_idx}: {command}")

# # Test classification network
# print("\nClassification network commands:")
# classification_commands = manager._generate_vm_commands('maskocr', 0)
# for vm_idx, command in classification_commands.items():
#     print(f"VM {vm_idx}: {command}")