from icd_validation.infer import summarize_report


def test_summarize_report_variants():
    report = {
        'explain': [
            {'declared': 'C18.9', 'predicted_top': 'C18.9', 'predicted_prob': 0.9, 'match_type': 'exact_match', 'score': 1.0},
            {'declared': 'C20.1', 'predicted_top': 'C20.5', 'predicted_prob': 0.2, 'match_type': 'same_category', 'score': 0.6},
            {'declared': 'Z99.9', 'predicted_top': None, 'predicted_prob': 0.0, 'match_type': 'mismatch', 'score': 0.0},
        ]
    }
    lines = summarize_report(report)
    assert len(lines) == 3
    assert 'Declared: C18.9' in lines[0]
    assert 'Decision: ACCEPT' in lines[0]
    assert 'Decision: REVIEW' in lines[1]
    assert 'Decision: REJECT' in lines[2]
