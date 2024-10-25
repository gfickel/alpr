"""
Detailed Explanation of Recursive Least Squares (RLS) Implementation for Plateau Detection

1. OVERVIEW & PURPOSE
--------------------
The implementation uses Recursive Least Squares (RLS) to fit a line to the data in real-time,
calculating both the slope (gradient) and our confidence in that slope. This is used to detect
when a training loss has plateaued.

2. THE MATHEMATICS
-----------------
The algorithm fits a line of the form: y = mx + b
Where:
- y is the loss value
- x is the position/time index
- m is the gradient (slope) we're trying to estimate
- b is the y-intercept

Example sequence: [0.5, 0.4, 0.3, 0.25, 0.24, 0.235, 0.233, 0.232]
When processed in reverse:
x: [0, 1, 2, 3,   4,   5,    6,    7]
y: [0.232, 0.233, 0.235, 0.24, 0.25, 0.3, 0.4, 0.5]

3. KEY COMPONENTS EXPLAINED
--------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import math

class RunningGradient:
    def __init__(self):
        self.clear()
    
    def clear(self):
        """
        Initialize/Reset the RLS estimator.
        
        R: 2x2 covariance matrix, initialized as identity * 1e6
           The large initial value (1e6) indicates high uncertainty
           [1e6  0  ]
           [0    1e6]
        
        w: 2x1 weight vector [gradient, intercept]
           [0]
           [0]
        """
        self.n = 0
        self.R = np.eye(2) * 1e6  # High initial uncertainty
        self.w = np.zeros((2, 1))  # Initial guess for parameters
        self.residual_squared = 0.0
    
    def add(self, y: float) -> None:
        """
        Add a new observation using RLS update equations.
        
        Example for first point (y=0.232):
        x = [0]  (first position)
            [1]  (bias term)
        """
        # Create feature vector [position, bias]
        x = np.array([[float(self.n)], [1.0]])
        
        # RLS Update equations:
        # 1. Calculate gain factor
        temp = 1.0 + float(x.T @ self.R @ x)
        tmp = self.R @ x
        
        # 2. Update covariance matrix
        self.R = self.R - (tmp @ tmp.T) / temp
        self.R = 0.5 * (self.R + self.R.T)  # Ensure symmetry
        
        # 3. Update weights based on prediction error
        prediction = float(x.T @ self.w)
        error = y - prediction
        self.w = self.w + self.R @ x * error
        
        # 4. Update squared residuals for standard error calculation
        self.residual_squared += error**2 * temp
        
        self.n += 1
    
    def gradient(self) -> float:
        """
        Return the current gradient estimate (first weight).
        The gradient tells us if the sequence is increasing or decreasing.
        """
        assert self.n > 1
        return float(self.w[0])
    
    def standard_error(self) -> float:
        """
        Calculate the standard error of the gradient estimate.
        This tells us how confident we are in our gradient estimate.
        
        Lower standard error = more confident
        Higher standard error = less confident
        """
        assert self.n > 2
        # Calculate variance of residuals
        s = self.residual_squared / (self.n - 2)
        # Adjust for the sequential nature of the data
        adjust = 12.0 / (self.n**3 - self.n)
        return np.sqrt(s * adjust)
    
    def probability_gradient_greater_than(self, threshold: float = 0) -> float:
        """
        Calculate probability that true gradient > threshold.
        Uses normal distribution properties:
        P(gradient > threshold) = 1 - Φ((threshold - estimate)/std_error)
        
        Returns:
        - Close to 1.0: Very confident gradient is greater than threshold
        - Close to 0.0: Very confident gradient is less than threshold
        - Close to 0.5: Uncertain
        """
        if self.n <= 2:
            return 0.5
        
        grad = self.gradient()
        stderr = self.standard_error()
        
        if stderr == 0:
            return 1.0 if grad > threshold else 0.0
        
        # Calculate z-score
        z_score = (grad - threshold) / stderr
        # Convert to probability using error function
        prob = 1.0 - 0.5 * math.erfc(-z_score / np.sqrt(2.0))
        
        return 1.0 - prob

# Example usage and visualization
def visualize_gradient_estimation():
    # Create a sequence that starts decreasing then plateaus
    x = np.linspace(0, 10, 50)
    y = 1.0 / (1 + np.exp(x - 5)) + np.random.normal(0, 0.01, 50)
    
    # Process sequence
    g = RunningGradient()
    gradients = []
    std_errors = []
    probs = []
    
    # Process in reverse order (as in plateau detection)
    for val in reversed(y):
        g.add(val)
        if g.n > 2:
            gradients.append(g.gradient())
            std_errors.append(g.standard_error())
            probs.append(g.probability_gradient_greater_than(0))
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot original sequence
    ax1.plot(x, y, 'b-', label='Values')
    ax1.set_title('Original Sequence')
    ax1.legend()
    
    # Plot gradient estimates
    ax2.plot(gradients, 'r-', label='Gradient')
    ax2.fill_between(range(len(gradients)), 
                    np.array(gradients) - np.array(std_errors), 
                    np.array(gradients) + np.array(std_errors), 
                    alpha=0.2, color='r')
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.set_title('Gradient Estimate with Standard Error')
    ax2.legend()
    
    # Plot probability of gradient > 0
    ax3.plot(probs, 'g-', label='P(gradient > 0)')
    ax3.axhline(y=0.51, color='k', linestyle='--', label='Decision Threshold')
    ax3.set_title('Probability of Positive Gradient')
    ax3.legend()
    
    plt.tight_layout()
    return fig

# Generate and display example
if __name__ == "__main__":
    fig = visualize_gradient_estimation()
    plt.show()
"""
4. PRACTICAL USAGE IN PLATEAU DETECTION
-------------------------------------
In practice, this is used to detect plateaus in training:

1. Process loss values in reverse order
2. For each point:
   - Update gradient estimate
   - Calculate probability of positive gradient
   - If probability < threshold (e.g., 0.51), count as plateau point

Example:
losses = [1.0, 0.8, 0.6, 0.5, 0.48, 0.47, 0.465, 0.463, 0.462]
- Early values show clear decrease
- Later values show plateau
- RLS will detect transition point
- Both gradient and confidence are considered

5. WHY IT WORKS WELL
-------------------
1. Recursive nature means O(1) updates
2. Maintains uncertainty estimates
3. Robust to noise through probabilistic approach
4. Adaptive to changing patterns
5. No need to store full history

The RLS approach is particularly good because it:
- Provides both estimate and uncertainty
- Updates efficiently with each new point
- Handles varying noise levels
- Can detect subtle changes in trend
"""
