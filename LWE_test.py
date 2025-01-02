import numpy as np

def generate_keys(n, q, sigma):
    s = np.random.randint(0, q, size=n)
    A = np.random.randint(0, q, size=(n, n))
    e = np.random.normal(0, sigma, size=n).astype(int) % q
    b = (A.dot(s) + e) % q
    return (A, b), s

def encrypt(public_key, message, q, t):
    A, b = public_key
    n = len(b)
    m = len(message)
    
    r = np.random.randint(0, 2, size=n)
    
    c1 = (r.dot(A)) % q
    c2 = (r.dot(b) + (q // t) * np.array(message)) % q
    
    return c1, c2

def decrypt(secret_key, ciphertext, q, t):
    s = secret_key
    c1, c2 = ciphertext
    
    m = c2 - c1.dot(s)
    m = m % q
    
    return np.round(t * m / q).astype(int) % t

# Example usage
n = 100  # Dimension
q = 40961  # Modulus (prime number)
sigma = 1  # Standard deviation for error
t = 16  # Number of possible values per symbol

# Generate keys
public_key, secret_key = generate_keys(n, q, sigma)

# Encrypt a message (each element can be 0 to t-1)
message = [3, 7, 12, 5, 0, 15, 2, 9]
ciphertext = encrypt(public_key, message, q, t)

# Decrypt the ciphertext
decrypted_message = decrypt(secret_key, ciphertext, q, t)

print(f"Original message: {message}")
print(f"Decrypted message: {decrypted_message.tolist()}")