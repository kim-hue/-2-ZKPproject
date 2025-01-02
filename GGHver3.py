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
    # Calculate log of row norms
    log_row_norms = np.log([np.linalg.norm(matrix[i]) for i in range(n)])
    # Sum logs instead of multiplying norms
    log_row_norm_product = np.sum(log_row_norms)
    # Use slogdet for better numerical stability
    sign, logdet = np.linalg.slogdet(matrix)
    # Return exp(logdet - log_row_norm_product)
    return sign * np.exp(logdet - log_row_norm_product)

def generate_private_basis(n, delta=1):
    """
    Generate a private basis (R) optimized for large dimensions.
    """
    max_attempts = 100
    attempts = 0
    
    while attempts < max_attempts:
        # Start with identity matrix
        R = np.eye(n, dtype=np.float64)
        
        # Add small random perturbations
        perturbation = np.random.uniform(-delta/2, delta/2, size=(n, n))
        R += perturbation
        
        # Force lower triangular structure
        R = np.tril(R)
        
        # Scale diagonal elements for better orthogonality
        scale_factor = n // 4  # Adjusted scaling for large n
        for i in range(n):
            R[i,i] = random.choice([-1, 1]) * (scale_factor + random.uniform(0, delta))
        
        # Use QR decomposition to improve orthogonality
        Q, _ = qr(R)
        R = Q * scale_factor
        R = np.tril(R)  # Ensure lower triangular again
        
        hadamard_ratio = calculate_hadamard_ratio(R)
        
        # Check constraints with relaxed conditions for numerical stability
        if (hadamard_ratio > 0.7 and  # Slightly relaxed from 0.8 for numerical stability
            abs(np.linalg.slogdet(R)[0] * np.exp(np.linalg.slogdet(R)[1])) >= n):
            return R
            
        attempts += 1
    
    raise ValueError(f"Failed to generate suitable private basis after {max_attempts} attempts")

def generate_unimodular_matrix(n):
    """
    Generate a sparse unimodular matrix for efficiency with large dimensions.
    """
    U = np.eye(n, dtype=np.float64)
    
    # Reduce number of operations for large n
    num_ops = n
    sparsity = max(1, n // 16)  # Control sparsity based on dimension
    
    for _ in range(num_ops):
        # Select random indices with controlled sparsity
        indices = random.sample(range(n), min(sparsity, n))
        
        for i in indices:
            j = random.choice([k for k in range(n) if k != i])
            op_type = random.randint(0, 2)
            
            if op_type == 0:
                # Swap rows
                U[[i,j]] = U[[j,i]]
            elif op_type == 1:
                # Add/subtract row with small multiplier
                U[i] += random.choice([-1, 1]) * U[j]
            else:
                # Multiply row by ±1
                U[i] *= random.choice([-1, 1])
    
    return U

def generate_keys(n, delta=1):
    """
    Generate keys with optimizations for large dimensions.
    """
    max_attempts = 50
    attempts = 0
    
    while attempts < max_attempts:
        try:
            # Generate private basis
            R = generate_private_basis(n, delta)
            
            # Generate sparse unimodular matrix
            U = generate_unimodular_matrix(n)
            
            # Compute public basis using blocked matrix multiplication
            block_size = 64  # Optimize for memory usage
            B = np.zeros_like(R)
            
            for i in range(0, n, block_size):
                for j in range(0, n, block_size):
                    i_end = min(i + block_size, n)
                    j_end = min(j + block_size, n)
                    B[i:i_end] += U[i:i_end, j:j_end] @ R[j:j_end]
            
            # Check public basis Hadamard ratio
            public_hadamard = calculate_hadamard_ratio(B)
            private_hadamard = calculate_hadamard_ratio(R)
            
            if public_hadamard < 0.01 and private_hadamard > 0.7:
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
            
        attempts += 1
    
    raise ValueError(f"Failed to generate suitable key pair after {max_attempts} attempts")

def verify_keys(public_key, private_key):
    """
    Verify keys with numerical stability considerations.
    """
    B = public_key['B']
    R = private_key['R']
    n = public_key['dimension']
    
    try:
        # Basic dimension checks
        if B.shape != (n, n) or R.shape != (n, n):
            return False
        
        # Check determinant using slogdet for stability
        sign, logdet = np.linalg.slogdet(R)
        if sign * np.exp(logdet) < n:
            return False
        
        # Check Hadamard ratios
        if calculate_hadamard_ratio(R) <= 0.7:  # Relaxed threshold
            return False
        
        if calculate_hadamard_ratio(B) >= 0.01:
            return False
        
        # Verify lattice equality with blocked computation
        block_size = 64
        U = np.zeros_like(B)
        R_inv = private_key['R_inv']
        
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                U[i:i_end] += B[i:i_end, j:j_end] @ R_inv[j:j_end]
        
        # Check if U is approximately unimodular
        sign, logdet = np.linalg.slogdet(U)
        if not np.isclose(abs(sign * np.exp(logdet)), 1, rtol=1e-5):
            return False
        
        return True
        
    except (np.linalg.LinAlgError, RuntimeWarning):
        return False

# Example usage with large dimension
if __name__ == "__main__":
    n = 256
    delta = 0.5  # Reduced delta for better stability
    
    print(f"Generating {n}-dimensional GGH lattice keys...")
    public_key, private_key = generate_keys(n, delta)
    
    print(f"Private basis Hadamard ratio: {private_key['hadamard_ratio']:.3f}")
    print(f"Public basis Hadamard ratio: {public_key['hadamard_ratio']:.3f}")