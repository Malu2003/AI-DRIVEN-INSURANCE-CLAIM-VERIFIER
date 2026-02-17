def test_build_label_list_and_multihot():
    from icd_validation.dataset import build_label_list, make_multihot

    samples = [ ("t1", ['A00','B01']), ("t2", ['A00']), ("t3", ['C18.9']) ]
    labels = build_label_list(samples, top_k=10)
    assert 'A00' in labels
    mh = make_multihot(['A00','C18.9'], labels)
    assert len(mh) == len(labels)
    # positions for present codes should be 1
    idx = {c:i for i,c in enumerate(labels)}
    if 'A00' in idx:
        assert mh[idx['A00']] == 1
