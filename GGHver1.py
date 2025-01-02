import numpy as np
from numpy.linalg import inv, det
import random

def generate_private_basis(n, delta=1):
    """
    Generate a private basis (R) for the GGH cryptosystem.
    
    Args:
        n (int): Dimension of the lattice
        delta (float): Parameter controlling the orthogonality of R
        
    Returns:
        numpy.ndarray: Private basis R
    """
    # Generate random upper triangular matrix
    R = np.tril(np.random.randint(-delta, delta + 1, size=(n, n)))
    
    # Make the diagonal entries larger to ensure good orthogonality
    for i in range(n):
        R[i,i] = random.choice([-1, 1]) * (delta * n)
    
    # Ensure matrix is invertible and has reasonable condition number
    while abs(det(R)) < n or np.linalg.cond(R) > n**2:
        R = np.tril(np.random.randint(-delta, delta + 1, size=(n, n)))
        for i in range(n):
            R[i,i] = random.choice([-1, 1]) * (delta * n)
    
    return R

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
    num_ops = n * 2
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
    Generate public and private keys for the GGH cryptosystem.
    
    Args:
        n (int): Dimension of the lattice
        delta (float): Parameter controlling the orthogonality
        
    Returns:
        tuple: (public_key, private_key)
    """
    # Generate private basis R
    R = generate_private_basis(n, delta)
    
    # Generate random unimodular matrix U
    U = generate_unimodular_matrix(n)
    
    # Compute public basis B = U * R
    B = U @ R
    
    # Store necessary key components
    private_key = {
        'R': R,
        'R_inv': inv(R),
        'dimension': n
    }
    
    public_key = {
        'B': B,
        'dimension': n
    }
    
    return public_key, private_key

def verify_keys(public_key, private_key):
    """
    Verify that the generated keys satisfy the required properties.
    
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
    
    # Check condition number of R
    if np.linalg.cond(R) > n**2:
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
    n = 64
    delta = 1
    
    public_key, private_key = generate_keys(n, delta)
    
    # Verify the generated keys
    is_valid = verify_keys(public_key, private_key)
    
    print(f"Generated {'valid' if is_valid else 'invalid'} keys for {n}-dimensional GGH lattice")
    print(f"Public basis condition number: {np.linalg.cond(public_key['B']):.2f}")
    print(f"Private basis condition number: {np.linalg.cond(private_key['R']):.2f}")