import subprocess
import concurrent.futures
import itertools
import os
import threading
import logging

def run_script(args, gpu_idx, cpu_semaphores):
    with cpu_semaphores[gpu_idx]:
        cmd = ["python", "maskocr_train.py"] + args + ["--device", f"{gpu_idx}"]
        env = os.environ.copy()
        # env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        # env["CUDA_LAUNCH_BLOCKING"] = '1'
        print(f"Running on GPU {gpu_idx}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)

def main():
    # Define the number of parallel executions
    N = 8  # Change this to your desired number of parallel executions

    cpu_semaphores = [threading.Semaphore(1) for _ in range(N)]

    # Define the argument combinations you want to try
    configurations = {
        # 'patch_size': ['32 8', '32 4', '32 16'],
        'batch_size': ['2048'],
        'embed_dim': ['384', '512', '256'],
        'num_heads': ['12', '8'],
        'num_encoder_layers': ['12', '8'],
        'num_decoder_layers': ['4', '2'],
        'max_sequence_length': ['7'],
        'dropout': ['0.1', '0.5'],
        'emb_dropout': ['0.1', '0.5'],
        'norm_image': ['1', '0'],
    }

    # Generate all combinations of arguments
    keys, values = zip(*configurations.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for combo in combinations:
        print(combo, '\n\n')
    print('num combinations', len(combinations))
    # exit(1)

    # Prepare argument lists for each combination
    arg_lists = []
    for idx,combo in enumerate(combinations):
        args = []
        for key, value in combo.items():
            args.extend([f'--{key}', value])
        args.extend(['--version', f'v{idx+1}'])
        arg_lists.append(args)

    # Run scripts in parallel with controlled CPU assignment
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
        futures = []
        for args in arg_lists:
            future = executor.submit(run_script, args, len(futures) % N, cpu_semaphores)
            futures.append(future)
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()