#!/usr/bin/env python3
import concurrent.futures
from hashlib import sha256
import multiprocessing as mp
import random

from rbloom import Bloom

bloom_filter = None

def hash_func(s):
    h = sha256(s.encode("utf-8")).digest()
    # use sys.byteorder instead of "big" for a small speedup when
    # reproducibility across machines isn't a concern
    return int.from_bytes(h[:16], "big", signed=True)

def test_bloom_fork():
    print("Entered test")
    all_numbers_to_test = list(range(101 * 500))
    random.shuffle(all_numbers_to_test)
    for number in all_numbers_to_test:
        _ = hash_func(str(number)) in bloom_filter

    numbers_to_test_true = [x * 500 for x in range(100)]
    random.shuffle(numbers_to_test_true)
    for number in numbers_to_test_true:
        _ = hash_func(str(number)) in bloom_filter


def main():
    global bloom_filter
    bloom_filter = Bloom(expected_items=100000, false_positive_rate=1e-3)
    for i in range(100):
        bloom_filter.add(hash_func(str(i * 500)))

    with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('fork')) as ex:
        futs = [ex.submit(test_bloom_fork) for _ in range(2)]
        for f in futs:
            f.result()
    print("All tests passed")


if __name__ == "__main__":
    main()
