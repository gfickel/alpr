import numpy as np
from typing import List, Union
import torch
# from scipy import stats
import math



class RunningGradient:
    """Maintains running statistics for gradient estimation using RLS."""
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Reset all statistics."""
        self.n = 0
        # Initialize R as 2x2 identity matrix * 1e6
        self.R = np.eye(2) * 1e6
        # Initialize w as 2x1 zero matrix
        self.w = np.zeros((2, 1))
        self.residual_squared = 0.0
    
    def add(self, y: float) -> None:
        """Add a new value using recursive least squares."""
        # Create x vector [n, 1]
        x = np.array([[float(self.n)], [1.0]])
        
        # Do recursive least squares computations
        temp = 1.0 + float(x.T @ self.R @ x)
        tmp = self.R @ x
        self.R = self.R - (tmp @ tmp.T) / temp
        # Ensure R stays symmetric for numerical stability
        self.R = 0.5 * (self.R + self.R.T)
        
        # Update weights
        self.w = self.w + self.R @ x * (y - float(x.T @ self.w))
        
        # Update residual
        self.residual_squared += (y - float(x.T @ self.w))**2 * temp
        
        self.n += 1
    
    def current_n(self) -> int:
        """Return the current number of points."""
        return self.n
    
    def gradient(self) -> float:
        """Get the current gradient estimate."""
        assert self.n > 1
        return float(self.w[0])
    
    def standard_error(self) -> float:
        """Calculate standard error of the gradient."""
        assert self.n > 2
        s = self.residual_squared / (self.n - 2)
        adjust = 12.0 / (self.n**3 - self.n)
        return np.sqrt(s * adjust)
    
    def probability_gradient_greater_than(self, threshold: float = 0) -> float:
        """
        Calculate the probability that the gradient is greater than the threshold.
        """
        if self.n <= 2:
            return 0.5
        
        grad = self.gradient()
        stderr = self.standard_error()
        
        if stderr == 0:
            return 1.0 if grad > threshold else 0.0
        
        z_score = (grad - threshold) / stderr
        prob = 1.0 - 0.5 * math.erfc(-z_score / np.sqrt(2.0))
        
        return 1.0 - prob  # Return probability gradient is greater than threshold



def find_upper_quantile(values: List[float], quantile: float) -> float:
    """Find the upper quantile value in a list of numbers."""
    if not values:
        return 0.0
    values_array = np.array(values, dtype=np.float64)  # Ensure float64 for precision
    return float(np.quantile(values_array, 1.0 - quantile))

def count_steps_without_decrease(
    losses: Union[List[float], torch.Tensor],
    probability_of_decrease: float = 0.51
) -> int:
    """
    Simple version of plateau detection without outlier removal.
    
    Args:
        losses: List of loss values or torch tensor containing loss history
        probability_of_decrease: Threshold probability for considering the loss as decreasing
        
    Returns:
        Number of steps since the last detected decrease in loss
    """
    assert 0.5 < probability_of_decrease < 1, "probability_of_decrease must be between 0.5 and 1"
    
    # Convert torch tensor to list if necessary
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy().tolist()
    
    if not losses:
        return 0
        
    # Initialize gradient calculator
    g = RunningGradient()
    count = 0
    
    # Iterate through losses in reverse order
    for j, loss in enumerate(reversed(losses), 1):
        g.add(float(loss))  # Ensure loss is float
        
        if g.current_n() > 2:
            # Check if we're confident the loss is still decreasing
            prob_decreasing = g.probability_gradient_greater_than(0)
            
            # Update count if we're not confident in the decrease
            if prob_decreasing < probability_of_decrease:
                count = j
                
    return count

def count_steps_without_decrease_robust(
    losses: Union[List[float], torch.Tensor],
    probability_of_decrease: float = 0.51,
    quantile_discard: float = 0.10
) -> int:
    """
    Robust version of plateau detection with outlier removal.
    
    Args:
        losses: List of loss values or torch tensor containing loss history
        probability_of_decrease: Threshold probability for considering the loss as decreasing
        quantile_discard: Fraction of highest values to discard as outliers
        
    Returns:
        Number of steps since the last detected decrease in loss
    """
    assert 0 <= quantile_discard <= 1, "quantile_discard must be between 0 and 1"
    assert 0.5 < probability_of_decrease < 1, "probability_of_decrease must be between 0.5 and 1"
    
    # Convert torch tensor to list if necessary
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy().tolist()
    
    if not losses:
        return 0
        
    # Find threshold for outlier removal
    quantile_thresh = find_upper_quantile(losses, quantile_discard)
    
    # Initialize gradient calculator
    g = RunningGradient()
    count = 0
    
    # Iterate through losses in reverse order
    for j, loss in enumerate(reversed(losses), 1):
        # Ignore values above the quantile threshold
        if loss <= quantile_thresh:
            g.add(float(loss))  # Ensure loss is float
            
        if g.current_n() > 2:
            # Check if we're confident the loss is still decreasing
            prob_decreasing = g.probability_gradient_greater_than(0)
            
            # Update count if we're not confident in the decrease
            if prob_decreasing < probability_of_decrease:
                count = j
                
    return count

def is_in_plateau(
    losses: Union[List[float], torch.Tensor],
    threshold: int,
    probability_of_decrease: float = 0.51,
    quantile_discard: float = 0.10
) -> bool:
    """
    Determine if the training is in a plateau by checking both methods.
    
    Args:
        losses: List of loss values or torch tensor containing loss history
        threshold: Number of steps without improvement to consider as plateau
        probability_of_decrease: Threshold probability for considering the loss as decreasing
        quantile_discard: Fraction of highest values to discard as outliers
        
    Returns:
        True if training is in plateau, False otherwise
    """
    steps_simple = count_steps_without_decrease(losses, probability_of_decrease)
    steps_robust = count_steps_without_decrease_robust(
        losses, 
        probability_of_decrease,
        quantile_discard
    )
    
    return steps_simple > threshold and steps_robust > threshold

# Example usage and tests:
if __name__ == "__main__":
    # Test Case 1: Clear plateau with some noise
    plateau_case = [0.5, 0.45, 0.42, 0.41, 0.405, 0.403, 0.402, 0.401, 0.4015, 0.402]
    
    # Test Case 2: Steady decrease (no plateau)
    decreasing_case = [0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    
    # Test Case 3: Plateau with outliers
    plateau_with_outliers = [0.5, 0.45, 0.42, 0.41, 0.405, 0.8, 0.403, 0.402, 0.9, 0.401]
    
    # Test Case 4: Noisy decrease
    noisy_decrease = [0.5, 0.47, 0.43, 0.44, 0.38, 0.39, 0.35, 0.33, 0.31, 0.29]
    
    # Test Case 5: Short sequence
    short_sequence = [0.5, 0.45, 0.44]
    
    # Parameters for testing
    threshold = 5
    prob_decrease = 0.51
    quantile_discard = 0.10
    
    # Function to run all tests on a single case
    def test_case(losses, case_name):
        steps_simple = count_steps_without_decrease(losses, prob_decrease)
        steps_robust = count_steps_without_decrease_robust(losses, prob_decrease, quantile_discard)
        in_plateau = is_in_plateau(losses, threshold, prob_decrease, quantile_discard)
        
        print(f"\nTest Case: {case_name}")
        print(f"Loss values: {[f'{x:.3f}' for x in losses]}")
        print(f"Steps without decrease (simple): {steps_simple}")
        print(f"Steps without decrease (robust): {steps_robust}")
        print(f"Is in plateau (threshold={threshold}): {in_plateau}")
        print("-" * 80)
    
    # Run all test cases
    test_cases = [
        (plateau_case, "Clear Plateau with Noise"),
        (decreasing_case, "Steady Decrease"),
        (plateau_with_outliers, "Plateau with Outliers"),
        (noisy_decrease, "Noisy Decrease"),
        (short_sequence, "Short Sequence"),
    ]
    
    print("Running Plateau Detection Tests")
    print("=" * 80)
    
    for losses, case_name in test_cases:
        test_case(losses, case_name)
    
    # Test with PyTorch tensors
    print("\nTesting with PyTorch tensors:")
    print("=" * 80)
    tensor_case = torch.tensor(plateau_case)
    test_case(tensor_case, "PyTorch Tensor Input")
    
    # Example of usage in training loop
    print("\nExample Training Loop Usage:")
    print("=" * 80)
    print("""
    # In your training loop:
    loss_history = []
    patience = 10
    
    for epoch in range(num_epochs):
        # Your training code...
        loss_history.append(float(current_loss))
        
        if epoch % check_frequency == 0:
            if is_in_plateau(loss_history, threshold=patience):
                # Reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print(f"Learning rate reduced at epoch {epoch}")
    """)