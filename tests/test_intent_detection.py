"""Tests for memory-type intent detection and decision classification."""

from agent_memory.ingest import classify_memory_type
from agent_memory.search import _detect_memory_type_intent, _query_mentions_memory_type


class TestQueryMentionsMemoryType:
    def test_lesson_keyword_matches_phrase(self):
        assert _query_mentions_memory_type("lesson learned", "lesson") is True

    def test_lesson_learning_keyword_currently_matches(self):
        """Current behavior: 'learning' is an accepted lesson keyword."""
        assert _query_mentions_memory_type("learning rate", "lesson") is True

    def test_lesson_hata_no_longer_matches(self):
        assert _query_mentions_memory_type("bu hata neden oluyor", "lesson") is False

    def test_lesson_ogren_matches(self):
        assert _query_mentions_memory_type("hatadan öğren", "lesson") is True

    def test_config_keyword_matches(self):
        assert _query_mentions_memory_type("config", "config_change") is True

    def test_ayar_keyword_matches(self):
        assert _query_mentions_memory_type("ayar", "config_change") is True

    def test_decision_keyword_matches(self):
        assert _query_mentions_memory_type("karar", "decision") is True

    def test_preference_keyword_matches(self):
        assert _query_mentions_memory_type("tercih", "preference") is True

    def test_rule_keyword_matches_turkish(self):
        assert _query_mentions_memory_type("kural", "rule") is True

    def test_rule_keyword_matches_english(self):
        assert _query_mentions_memory_type("policy", "rule") is True

    def test_empty_query_matches_nothing(self):
        for memory_type in ("lesson", "config_change", "decision", "preference", "rule"):
            assert _query_mentions_memory_type("", memory_type) is False

    def test_word_boundary_prevents_substring_false_positive(self):
        assert _query_mentions_memory_type("misconfigured", "config_change") is False


class TestDetectMemoryTypeIntent:
    def test_detects_lesson(self):
        assert _detect_memory_type_intent("behavioral lessons") == "lesson"

    def test_detects_config_change(self):
        assert _detect_memory_type_intent("config değişikliği") == "config_change"

    def test_detects_decision(self):
        assert _detect_memory_type_intent("karar alındı") == "decision"

    def test_detects_rule(self):
        assert _detect_memory_type_intent("kural nedir") == "rule"

    def test_detects_preference(self):
        assert _detect_memory_type_intent("tercih ediyorum") == "preference"

    def test_returns_none_for_normal_query(self):
        assert _detect_memory_type_intent("normal query without type keywords") is None

    def test_returns_none_for_removed_hata_signal(self):
        assert _detect_memory_type_intent("son hata mesajı") is None

    def test_returns_none_for_empty_query(self):
        assert _detect_memory_type_intent("") is None


class TestClassifyMemoryTypeDecision:
    def test_decision_bracket_marker(self):
        assert classify_memory_type("[Decision] tamam yapalım") == "decision"

    def test_decision_turkish_marker(self):
        assert classify_memory_type("KARAR: bunu yapacağız") == "decision"

    def test_non_decision_text_is_not_decision(self):
        assert classify_memory_type("Bunu sonra konuşuruz") != "decision"
