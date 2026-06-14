from scroot.text_utils import split_sentences, extract_claims


def test_split_sentences_basic():
    text = "This is a sentence. This is another one. And a third."
    sentences = split_sentences(text)
    assert len(sentences) >= 2


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_none_like():
    assert split_sentences("") == []


def test_split_sentences_single():
    result = split_sentences("Hello world")
    assert result == ["Hello world"]


def test_extract_claims_filters_greetings():
    text = "Hi there! We offer a 30-day refund. Thanks for asking."
    claims = extract_claims(text)
    assert not any(c.lower().startswith("hi") for c in claims)


def test_extract_claims_filters_questions():
    text = "What is your name? My name is Claude. I am an AI."
    claims = extract_claims(text)
    assert not any(c.endswith("?") for c in claims)


def test_extract_claims_min_words():
    text = "Yes. No. We offer a 30-day full refund at no extra cost."
    claims = extract_claims(text)
    assert all(len(c.split()) >= 3 for c in claims)


def test_extract_claims_empty():
    assert extract_claims("") == []


def test_extract_claims_real_text():
    text = "We offer a 30-day full refund at no extra cost. You can return any item within 30 days of purchase."
    claims = extract_claims(text)
    assert len(claims) >= 1
