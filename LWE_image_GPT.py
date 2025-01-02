import numpy as np
from PIL import Image

# Parameters
n, m, q = 16, 16, 40961  # Example parameters
A = np.random.randint(0, q, (n, m))
s = np.random.randint(0, q, n)
e = np.random.randint(-10, 11, m)
b = (A.T @ s + e) % q
public_key = (A, b)
secret_key = s

# Encrypt function
def encrypt(public_key, pixel_values):
    A, b = public_key
    encrypted_pixels = []
    for pixel in pixel_values:
        x = np.random.randint(0, 2, A.shape[1])  # random binary vector
        u = (A.T @ x) % q
        v = (b.T @ x + pixel) % q
        encrypted_pixels.append((u, v))
    return encrypted_pixels

# Decrypt function
def decrypt(secret_key, ciphertext):
    decrypted_pixels = []
    for u, v in ciphertext:
        decrypted_value = (v - np.dot(u, secret_key)) % q
        decrypted_pixels.append(decrypted_value if decrypted_value < 128 else decrypted_value - q)
    return np.array(decrypted_pixels)

# Load and prepare the image
img = Image.open('F:/[2]ZKPproject/bit-256-x-256-Grayscale-Lena-Image_Q320.jpg').convert('L')  # Load as grayscale
img = img.resize((256, 256))  # Ensure the image is 256x256
pixel_values = np.array(img).flatten()  # Flatten the image into a 1D array

# Encrypt the pixel values
ciphertext = encrypt(public_key, pixel_values)

# Prepare to save the encrypted image
# For simplification, we'll save only the 'v' values (one way to represent the encrypted image)
encrypted_image_data = np.array([v for _, v in ciphertext], dtype=np.uint16)  # Use uint16 because encrypted values may exceed 255

# Reshape back to the 256x256 format
encrypted_image_data = encrypted_image_data.reshape((256, 256))

# Save the encrypted image
encrypted_image = Image.fromarray(encrypted_image_data)
encrypted_image.save('encrypted_image.jpg')

print("Image encrypted and saved as 'encrypted_image.jpg'.")