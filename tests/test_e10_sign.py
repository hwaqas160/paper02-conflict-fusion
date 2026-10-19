from src.e10_minimal import sign_test_p


def test_sign_test_extremes():
    assert abs(sign_test_p(13, 13) - 2 / 8192) < 1e-12
    assert sign_test_p(0, 13) == sign_test_p(13, 13)


def test_sign_test_matches_reported_value():
    assert abs(sign_test_p(10, 13) - 0.09228515625) < 1e-12


def test_sign_test_symmetric_and_bounded():
    for n in (1, 5, 12, 230):
        for w in range(n + 1):
            p = sign_test_p(w, n)
            assert 0.0 < p <= 1.0
            assert abs(p - sign_test_p(n - w, n)) < 1e-12


def test_sign_test_no_data():
    assert sign_test_p(0, 0) == 1.0
