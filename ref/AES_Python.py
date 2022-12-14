from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import time

class AES_Python:

    # encrypt(self, plaintext):
    #  Encrypt the plaintext and return the ciphertext.
    #  Inputs:
    #   - plaintext: hex string containing the plaintext
    #  Output:
    #   - hex string containing the ciphertext
    #   - Recorded time to perform the operation

    def encrypt(self, plaintext, cipherkey):
        start = time.time()
        # Add padding if the plaintext does not contain a multiple of 16 bytes (= 32 hex numbers)
        if len(plaintext) % 32:
            padding = ''.join('0' for n in range(32 - len(plaintext) % 32))
            plaintext += padding
        input = bytes.fromhex(plaintext)
        cipher = Cipher(algorithms.AES(bytes.fromhex(cipherkey)), modes.ECB())
        encryptor = cipher.encryptor()
        ct = encryptor.update(input) + encryptor.finalize()
        end = time.time()
        return ct.hex(), end - start
    
    # decrypt(self, ciphertext):
    #  Decrypt the ciphertext and return the plaintext.
    #  Inputs:
    #   - ciphertext: hex string containing the ciphertext
    #  Output:
    #   - hex string containing the plaintext
    #   - Recorded time to perform the operation
    def decrypt(self, ciphertext, cipherkey):
        start = time.time()
        input = bytes.fromhex(ciphertext)
        cipher = Cipher(algorithms.AES(bytes.fromhex(cipherkey)), modes.ECB())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(input) + decryptor.finalize()
        end = time.time()
        return plaintext.hex(), end - start

def test_AES_Python():
    hex_key = "000102030405060708090a0b0c0d0e0f"
    hex_input = "00112233445566778899aabbccddeeff"
    aes = AES_Python()
    ciphertext = aes.encrypt(hex_input, hex_key)[0]
    plaintext = aes.decrypt(ciphertext, hex_key)[0]
    assert ciphertext == "69c4e0d86a7b0430d8cdb78070b4c55a"
    assert plaintext == hex_input

def test2_AES_Python():
    hex_key = "000102030405060708090a0b0c0d0e0f"
    hex_input = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
    aes = AES_Python()
    ciphertext = aes.encrypt(hex_input, hex_key)[0]
    plaintext = aes.decrypt(ciphertext, hex_key)[0]
    assert ciphertext == "69c4e0d86a7b0430d8cdb78070b4c55a69c4e0d86a7b0430d8cdb78070b4c55a"
    assert plaintext == hex_input

def test3_AES_Python():
    hex_key = "000102030405060708090a0b0c0d0e0f"
    hex_input = "00112233445566778899aabbccddeeff0011223344556677"
    aes = AES_Python()
    ciphertext = aes.encrypt(hex_input, hex_key)[0]
    plaintext = aes.decrypt(ciphertext, hex_key)[0]
    assert ciphertext == "69c4e0d86a7b0430d8cdb78070b4c55ab61b9091935d3ee92634dcd834779663"
    assert plaintext == "00112233445566778899aabbccddeeff00112233445566770000000000000000"
