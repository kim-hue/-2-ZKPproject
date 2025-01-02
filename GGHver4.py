import numpy as np
from numpy.linalg import inv, det
import random
from scipy.linalg import qr
import warnings
warnings.filterwarnings('ignore')

def calculate_hadamard_ratio(matrix):
    """
    Calculate the Hadamard ratio of a matrix using log space for numerical stability.
    """
    n = matrix.shape[0]
    log_row_norms = np.log([np.linalg.norm(matrix[i]) for i in range(n)])
    log_row_norm_product = np.sum(log_row_norms)
    sign, logdet = np.linalg.slogdet(matrix)
    return sign * np.exp(logdet - log_row_norm_product)

def generate_private_basis(n, delta=1):
    """
    Generate a private basis (R) optimized for large dimensions.
    Uses dynamic parameter adjustment for better success rate.
    """
    max_attempts = 100
    attempts = 0
    
    # Dynamic parameter adjustment based on dimension
    initial_scale = np.sqrt(n)
    orthogonality_target = max(0.6, 0.8 - (n/1000))  # Relaxed target for large n
    
    while attempts < max_attempts:
        try:
            # Generate a random orthogonal matrix using QR decomposition
            A = np.random.normal(0, 1, (n, n))
            Q, _ = qr(A)
            
            # Scale the orthogonal matrix
            R = Q * initial_scale
            
            # Add controlled perturbations to lower triangular part
            perturbation = np.tril(np.random.uniform(-delta, delta, (n, n)))
            R = np.tril(R + perturbation)
            
            # Enhance diagonal dominance
            for i in range(n):
                # Scale diagonal elements based on dimension
                diag_scale = initial_scale * (1 + np.random.uniform(0, delta))
                R[i,i] = random.choice([-1, 1]) * diag_scale
            
            hadamard_ratio = calculate_hadamard_ratio(R)
            
            # Check constraints with dimension-adjusted thresholds
            if (hadamard_ratio > orthogonality_target and 
                abs(np.linalg.slogdet(R)[0] * np.exp(np.linalg.slogdet(R)[1])) >= n/2):
                return R
                
        except (np.linalg.LinAlgError, RuntimeWarning):
            pass
            
        attempts += 1
        
        # Adaptive parameter adjustment
        if attempts % 20 == 0:
            initial_scale *= 1.2
            delta *= 0.9
    
    raise ValueError(f"Failed to generate suitable private basis. Try adjusting delta or relaxing constraints.")

def generate_unimodular_matrix(n):
    """
    Generate a sparse unimodular matrix optimized for large dimensions.
    """
    U = np.eye(n, dtype=np.float64)
    
    # Reduce operations for large matrices
    num_ops = max(n//2, 50)
    sparsity = max(2, n//32)
    
    for _ in range(num_ops):
        indices = random.sample(range(n), min(sparsity, n))
        
        for i in indices:
            j = random.choice([k for k in range(max(0, i-10), min(n, i+11)) if k != i])
            op_type = random.randint(0, 1)
            
            if op_type == 0 and abs(i-j) < 5:  # Local swaps only
                U[[i,j]] = U[[j,i]]
            else:
                # Small integer multipliers for better stability
                U[i] += random.choice([-1, 1]) * U[j]
    
    return U

def generate_keys(n, delta=1):
    """
    Generate keys with improved parameter handling and error recovery.
    """
    max_attempts = 50
    attempts = 0
    
    # Adaptive delta based on dimension
    adaptive_delta = delta * (1.0 / np.sqrt(n/4))
    
    while attempts < max_attempts:
        try:
            # Generate private basis with adapted parameters
            R = generate_private_basis(n, adaptive_delta)
            
            # Generate unimodular matrix
            U = generate_unimodular_matrix(n)
            
            # Compute public basis with blocked multiplication
            block_size = min(64, n//4)
            B = np.zeros_like(R)
            
            for i in range(0, n, block_size):
                for j in range(0, n, block_size):
                    i_end = min(i + block_size, n)
                    j_end = min(j + block_size, n)
                    B[i:i_end] += U[i:i_end, j:j_end] @ R[j:j_end]
            
            # Verify Hadamard ratios
            public_hadamard = calculate_hadamard_ratio(B)
            private_hadamard = calculate_hadamard_ratio(R)
            
            # Adjust thresholds based on dimension
            private_threshold = max(0.6, 0.8 - (n/1000))
            public_threshold = min(0.01, 0.01 + (n/10000))
            
            if public_hadamard < public_threshold and private_hadamard > private_threshold:
                private_key = {
                    'R': R,
                    'R_inv': inv(R),
                    'dimension': n,
                    'hadamard_ratio': private_hadamard
                }
                
                public_key = {
                    'B': B,
                    'dimension': n,
                    'hadamard_ratio': public_hadamard
                }
                
                if verify_keys(public_key, private_key):
                    return public_key, private_key
                    
        except (np.linalg.LinAlgError, RuntimeWarning):
            pass
            
        # Adaptive parameter adjustment
        attempts += 1
        adaptive_delta *= 0.95
    
    raise ValueError(f"Failed to generate suitable key pair. Try adjusting initial delta value.")

def verify_keys(public_key, private_key):
    """
    Verify keys with relaxed constraints for large dimensions.
    """
    B = public_key['B']
    R = private_key['R']
    n = public_key['dimension']
    
    try:
        # Dimension-adjusted thresholds
        private_threshold = max(0.6, 0.8 - (n/1000))
        public_threshold = min(0.01, 0.01 + (n/10000))
        
        # Basic checks
        if calculate_hadamard_ratio(R) <= private_threshold:
            return False
        if calculate_hadamard_ratio(B) >= public_threshold:
            return False
            
        # Verify lattice equality with relaxed tolerance
        R_inv = private_key['R_inv']
        U = B @ R_inv
        
        # Check unimodular property with dimension-appropriate tolerance
        sign, logdet = np.linalg.slogdet(U)
        return np.isclose(abs(sign * np.exp(logdet)), 1, rtol=1e-4 * np.sqrt(n))
        
    except (np.linalg.LinAlgError, RuntimeWarning):
        return False

# Example usage
if __name__ == "__main__":
    n = 64
    initial_delta = 0.3  # Reduced initial delta for n=256
    
    print(f"Generating {n}-dimensional GGH lattice keys...")
    try:
        public_key, private_key = generate_keys(n, initial_delta)
        print(f"Successfully generated keys:")
        print(f"Private basis Hadamard ratio: {private_key['hadamard_ratio']:.3f}")
        print(f"Public basis Hadamard ratio: {public_key['hadamard_ratio']:.3f}")
    except ValueError as e:
        print(f"Error: {e}")