import random
from tqdm import tqdm

lengths = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 4*268435456]
# lengths = [4*268435456]
for length in lengths:
    s = ''.join(random.choice('0123456789abcdef') for _ in tqdm(range(length)))

    with open(f"test_case_{length}.txt", "w") as f:
        f.write(s)