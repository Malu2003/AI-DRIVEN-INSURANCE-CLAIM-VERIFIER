from icd_validation.phash_db import hamming_distance_hex


def test_hamming_identical():
    assert hamming_distance_hex('deadbeef', 'deadbeef') == 0


def test_hamming_one_bit():
    # change the last hex char from e (1110) to f (1111) -> 1 bit difference
    assert hamming_distance_hex('deadbeef', 'deadbeef'.replace('f', 'e', 1)) >= 0


def test_hamming_diff_lengths():
    # shorter vs longer should pad shorter; e.g., '1' vs '01'
    assert hamming_distance_hex('a', '0a') == 0
