import random

lengths = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]
for length in lengths:
    string = ''.join(random.choice('0123456789abcdef') for n in range(length))
    file = open(f"test_case_{length}.txt", "w")
    file.write(string)
    file.close()
