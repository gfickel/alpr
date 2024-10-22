import torch
import argparse
import time

def main(device_num):
    # Set the device
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define matrix size
    matrix_size = 30000

    iteration = 0
    try:
        while True:
            iteration += 1
            # print(f"\nIteration {iteration}")

            # Create large matrices
            matrix1 = torch.randn(matrix_size, matrix_size, device=device)
            matrix2 = torch.randn(matrix_size, matrix_size, device=device)

            # print(f"Created two {matrix_size}x{matrix_size} matrices")

            # Perform matrix multiplication
            # print("Starting matrix multiplication...")
            result = torch.matmul(matrix1, matrix2)
            # print("Matrix multiplication completed")

            # Perform some element-wise operations
            # print("Performing element-wise operations...")
            result = torch.sin(result)
            result = torch.exp(result)
            # print("Element-wise operations completed")

            # Calculate and print some statistics
            # print(f"Result matrix shape: {result.shape}")
            # print(f"Result matrix mean: {result.mean().item():.4f}")
            # print(f"Result matrix std: {result.std().item():.4f}")

            # Clear GPU memory
            # del matrix1, matrix2, result
            # torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nScript terminated by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyTorch operations continuously on a specified GPU")
    parser.add_argument("--device", type=int, help="GPU device number to use")
    args = parser.parse_args()

    main(args.device)