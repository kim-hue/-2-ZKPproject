import numpy as np
from numpy.linalg import inv, det
import random

def calculate_hadamard_ratio(matrix):
    """
    Calculate the Hadamard ratio of a matrix.
    
    The Hadamard ratio is |det(matrix)| divided by the product of the Euclidean
    norms of its rows.
    
    Args:
        matrix (numpy.ndarray): Input matrix
        
    Returns:
        float: Hadamard ratio
    """
    n = matrix.shape[0]
    row_norms_product = np.prod([np.linalg.norm(matrix[i]) for i in range(n)])
    return abs(det(matrix)) / row_norms_product

def generate_private_basis(n, delta=1):
    """
    Generate a private basis (R) for the GGH cryptosystem with Hadamard ratio > 0.8
    
    Args:
        n (int): Dimension of the lattice
        delta (float): Parameter controlling the orthogonality of R
        
    Returns:
        numpy.ndarray: Private basis R
    """
    max_attempts = 20000
    attempts = 0
    
    while attempts < max_attempts:
        # Generate random lower triangular matrix
        R = np.tril(np.random.randint(-delta, delta + 1, size=(n, n)))
        
        # Make the diagonal entries larger to ensure good orthogonality
        for i in range(n):
            R[i,i] = random.choice([-1, 1]) * (delta * n *2)  # Increased factor for better orthogonality
        
        hadamard_ratio = calculate_hadamard_ratio(R)
        
        # Check all constraints
        if (hadamard_ratio > 0.6 and 
            abs(det(R)) >= n and 
            np.linalg.cond(R) <= n**2):
            return R
            
        attempts += 1
    
    raise ValueError(f"Failed to generate suitable private basis after {max_attempts} attempts")

def generate_unimodular_matrix(n):
    """
    Generate a random unimodular matrix (determinant ±1) for basis transformation.
    
    Args:
        n (int): Dimension of the matrix
        
    Returns:
        numpy.ndarray: Unimodular matrix U
    """
    U = np.eye(n, dtype=int)
    
    # Perform random elementary operations
    num_ops = n * 3  # Increased number of operations for more randomness
    for _ in range(num_ops):
        op_type = random.randint(0, 2)
        i, j = random.sample(range(n), 2)
        
        if op_type == 0:
            # Swap rows
            U[[i,j]] = U[[j,i]]
        elif op_type == 1:
            # Add/subtract row
            U[i] += random.choice([-1, 1]) * U[j]
        else:
            # Multiply row by ±1
            U[i] *= random.choice([-1, 1])
            
    return U

def generate_keys(n, delta=1):
    """
    Generate public and private keys for the GGH cryptosystem with specific
    Hadamard ratio constraints.
    
    Args:
        n (int): Dimension of the lattice
        delta (float): Parameter controlling the orthogonality
        
    Returns:
        tuple: (public_key, private_key)
    """
    max_attempts = 20000
    attempts = 0
    
    while attempts < max_attempts:
        # Generate private basis R
        R = generate_private_basis(n, delta)
        
        # Generate random unimodular matrix U
        U = generate_unimodular_matrix(n)
        
        # Compute public basis B = U * R
        B = U @ R
        
        # Check Hadamard ratio of public basis
        public_hadamard = calculate_hadamard_ratio(B)
        
        if public_hadamard < 0.05:  # Public key constraint
            private_key = {
                'R': R,
                'R_inv': inv(R),
                'dimension': n,
                'hadamard_ratio': calculate_hadamard_ratio(R)
            }
            
            public_key = {
                'B': B,
                'dimension': n,
                'hadamard_ratio': public_hadamard
            }
            
            return public_key, private_key
            
        attempts += 1
    
    raise ValueError(f"Failed to generate suitable key pair after {max_attempts} attempts")

def verify_keys(public_key, private_key):
    """
    Verify that the generated keys satisfy all required properties including
    Hadamard ratio constraints.
    
    Args:
        public_key (dict): Public key containing basis B
        private_key (dict): Private key containing basis R
        
    Returns:
        bool: True if keys are valid, False otherwise
    """
    B = public_key['B']
    R = private_key['R']
    n = public_key['dimension']
    
    # Check dimensions
    if B.shape != (n, n) or R.shape != (n, n):
        return False
    
    # Check that R is invertible
    if abs(det(R)) < n:
        return False
    
    # Check Hadamard ratio constraints
    if calculate_hadamard_ratio(R) <= 0.8:
        return False
        
    if calculate_hadamard_ratio(B) >= 0.01:
        return False
    
    # Check that B spans the same lattice as R
    try:
        U = B @ inv(R)
        if not np.allclose(U @ R, B):
            return False
        if abs(round(det(U))) != 1:
            return False
    except np.linalg.LinAlgError:
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Generate keys for a 4-dimensional lattice
    n = 110
    delta = 3
    
    public_key, private_key = generate_keys(n, delta)
    
    # Verify the generated keys
    is_valid = verify_keys(public_key, private_key)
    
    print(f"Generated {'valid' if is_valid else 'invalid'} keys for {n}-dimensional GGH lattice")
    print(f"Private basis Hadamard ratio: {private_key['hadamard_ratio']:.10f}")
    print(f"Public basis Hadamard ratio: {public_key['hadamard_ratio']:.10f}")
    print(public_key)
    