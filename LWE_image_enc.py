import numpy as np
from PIL import Image

def generate_keys(n, q, sigma):
    s = np.random.randint(0, q, size=n)
    A = np.random.randint(0, q, size=(n, n))
    e = np.random.normal(0, sigma, size=n).astype(int) % q
    b = (A.dot(s) + e) % q
    return (A, b), s

def encrypt(public_key, message, q, t):
    A, b = public_key
    n = len(b)
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

def encrypt_image(public_key, image_array, q, t, block_size):
    height, width = image_array.shape
    encrypted_image = np.zeros((height, width * 2), dtype=np.uint8)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_array[i:i+block_size, j:j+block_size].flatten()
            c1, c2 = encrypt(public_key, block, q, t)
            
            c1_mapped = (c1 % 256).astype(np.uint8)
            c2_mapped = (c2 % 256).astype(np.uint8)
            
            encrypted_image[i:i+block_size, j*2:j*2+block_size] = c1_mapped.reshape((block_size, block_size))
            encrypted_image[i:i+block_size, j*2+block_size:(j+1)*2*block_size] = c2_mapped.reshape((block_size, block_size))
    
    return encrypted_image

def decrypt_image(secret_key, encrypted_image, q, t, block_size):
    height, width = encrypted_image.shape
    width = width // 2
    decrypted_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            c1 = encrypted_image[i:i+block_size, j*2:j*2+block_size].flatten()
            c2 = encrypted_image[i:i+block_size, j*2+block_size:(j+1)*2*block_size].flatten()
            
            c1_original = c1.astype(np.int64)
            c2_original = c2.astype(np.int64)
            
            decrypted_block = decrypt(secret_key, (c1_original, c2_original), q, t)
            decrypted_image[i:i+block_size, j:j+block_size] = decrypted_block.reshape((block_size, block_size))
    
    return decrypted_image

# Parameters
n = 1024
q = 40961
sigma = 1
t = 256
block_size = 16

# Generate keys
public_key, secret_key = generate_keys(n, q, sigma)

# Load and preprocess the image
image_path = "F:/[2]ZKPproject/bit-256-x-256-Grayscale-Lena-Image_Q320.jpg"  # Replace with your 256x256 JPG image path
image = Image.open(image_path).convert('L').resize((256, 256))
image_array = np.array(image)

# Encrypt the image
encrypted_image = encrypt_image(public_key, image_array, q, t, block_size)

# Save the encrypted image
encrypted_image_path = "encrypted_image.jpg"
encrypted_image_pil = Image.fromarray(encrypted_image.astype(np.uint8))
encrypted_image_pil.save(encrypted_image_path, "JPEG")

# Decrypt the image
decrypted_image = decrypt_image(secret_key, encrypted_image, q, t, block_size)

# Save the decrypted image
decrypted_image_path = "decrypted_image.jpg"
decrypted_image_pil = Image.fromarray(decrypted_image.astype(np.uint8))
decrypted_image_pil.save(decrypted_image_path, "JPEG")

print("Encryption and decryption completed.")
print("Encrypted image saved as:", encrypted_image_path)
print("Decrypted image saved as:", decrypted_image_path)

