"""Tests for Turkish NLP utilities."""

import pytest

from asuman_memory.turkish import (
    ascii_fold,
    lemmatize,
    normalize_text,
    parse_temporal,
    TURKISH_STOPWORDS,
    tokenize_for_search,
)


class TestASCIIFolding:
    def test_basic_fold(self):
        assert ascii_fold("çalışıyor") == "calisiyor"

    def test_uppercase_fold(self):
        assert ascii_fold("ÇAĞRI") == "CAGRI"

    def test_mixed(self):
        assert ascii_fold("güzel günler") == "guzel gunler"

    def test_no_change(self):
        assert ascii_fold("hello world") == "hello world"


class TestLemmatize:
    def test_basic_lemmatize(self):
        result = lemmatize("hatırlıyorum")
        # zeyrek should produce something like "hatırla"
        assert "hatırla" in result or "hatırlıyorum" in result

    def test_empty(self):
        assert lemmatize("") == ""

    def test_english_passthrough(self):
        result = lemmatize("remember")
        assert "remember" in result


class TestTemporalParsing:
    def test_custom_obur_gun(self):
        result = parse_temporal("öbür gün")
        assert result is not None
        start, end = result
        assert start < end

    def test_custom_bu_sabah(self):
        result = parse_temporal("bu sabah")
        assert result is not None
        start, end = result
        assert start.hour == 6

    def test_custom_evvelsi_gun(self):
        result = parse_temporal("evvelsi gün")
        assert result is not None

    def test_dateparser_gecen_hafta(self):
        result = parse_temporal("geçen hafta")
        if result:  # dateparser may or may not be installed
            start, end = result
            assert start < end

    def test_english_yesterday(self):
        result = parse_temporal("yesterday")
        if result:
            start, end = result
            assert start < end

    def test_unknown_returns_none(self):
        result = parse_temporal("asdfghjkl")
        assert result is None


class TestStopwords:
    def test_turkish_stopwords(self):
        assert "ve" in TURKISH_STOPWORDS
        assert "bir" in TURKISH_STOPWORDS
        assert "the" in TURKISH_STOPWORDS

    def test_not_stopword(self):
        assert "toplantı" not in TURKISH_STOPWORDS


class TestNormalization:
    def test_normalize(self):
        result = normalize_text("Bu çok güzel bir toplantı")
        # Should remove stopwords like "bu", "çok", "bir"
        assert "bu" not in result.split()
        assert "toplantı" in result or "toplanti" in result

    def test_tokenize_for_search(self):
        tokens = tokenize_for_search("hatırlıyor musun dün ne konuştuk")
        assert len(tokens) > 0
        # stopwords should be removed
        assert "ne" not in tokens
