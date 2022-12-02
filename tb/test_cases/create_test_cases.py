import random

lengths = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456]
for length in lengths:
    string = ''.join(random.choice('0123456789abcdef') for n in range(length))
    file = open(f"test_case_{length}.txt", "w")
    file.write(string)
    file.close()
