from advisor.analysis.news_scorer import lexicon_score


def test_positive_headline_is_positive():
    assert lexicon_score("Earnings beat estimates, record profit") > 0


def test_negative_headline_is_negative():
    assert lexicon_score("Guidance cut, probe opened, lawsuit filed") < 0


def test_neutral_headline_is_zero():
    assert lexicon_score("Company schedules annual shareholder meeting") == 0.0


def test_score_is_bounded():
    assert -1.0 <= lexicon_score("beat beat beat surge upgrade") <= 1.0
    assert -1.0 <= lexicon_score("miss cut probe fraud recall") <= 1.0
