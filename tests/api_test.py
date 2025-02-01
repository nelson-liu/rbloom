#!/usr/bin/env python3
from hashlib import sha256
from rbloom import Bloom
import os


def hash_func(s):
    h = sha256(s.encode("utf-8")).digest()
    # use sys.byteorder instead of "big" for a small speedup when
    # reproducibility across machines isn't a concern
    return int.from_bytes(h[:16], "big", signed=True)


def test_bloom(bloom: Bloom):
    assert not bloom
    assert bloom.approx_items == 0.0

    bloom.add(hash_func('foo'))
    assert bloom
    assert bloom.approx_items > 0.0

    bloom.add(hash_func('bar'))

    assert hash_func('foo') in bloom
    assert hash_func('bar') in bloom
    assert hash_func('baz') not in bloom

    bloom.update(map(hash_func, ['baz', 'qux']))
    assert hash_func('baz') in bloom
    assert hash_func('qux') in bloom

    other = bloom.copy()
    assert other == bloom
    assert other is not bloom

    other.clear()
    assert not other
    assert other.approx_items == 0.0

    other.update(map(hash_func, ['foo', 'bar', 'baz', 'qux']))
    assert other == bloom

    other.update(hash_func(str(i*500)) for i in range(100000))
    for i in range(100000):
        assert hash_func(str(i*500)) in other
    assert bloom != other
    assert bloom & other == bloom
    assert bloom | other == other

    bloom &= other
    assert bloom < other

    orig = bloom.copy()
    bloom |= other
    assert bloom == other
    assert bloom > orig
    assert bloom >= orig
    assert bloom.issuperset(other)
    assert orig <= bloom
    assert orig.issubset(bloom)
    assert bloom >= bloom
    assert bloom.issuperset(bloom)
    assert bloom <= bloom
    assert bloom.issubset(bloom)

    bloom = orig.copy()
    bloom.update(other)
    assert bloom == other
    assert bloom > orig

    bloom = orig.copy()
    assert other == bloom.union(other)
    assert bloom == bloom.intersection(other)

    bloom.intersection_update(other)
    assert bloom == orig

    # TEST PERSISTENCE
    # find a filename that doesn't exist
    i = 0
    while os.path.exists(f'UNIT_TEST_{i}.bloom'):
        i += 1
    filename = f'test{i}.bloom'

    try:
        # save and load
        bloom.save(filename)
        bloom2 = Bloom.load(filename)
        assert bloom == bloom2
    finally:
        # remove the file
        os.remove(filename)

    # TEST bytes PERSISTENCE
    bloom_bytes = bloom.save_bytes()
    assert type(bloom_bytes) == bytes
    bloom3 = Bloom.load_bytes(bloom_bytes)
    assert bloom == bloom3


def api_suite():
    assert repr(Bloom(27_000, 0.0317)) == "<Bloom size_in_bits=193960 approx_items=0.0>"
    test_bloom(Bloom(13242, 0.0000001))
    test_bloom(Bloom(2837, 0.5))
    print('All API tests passed')


if __name__ == '__main__':
    api_suite()
