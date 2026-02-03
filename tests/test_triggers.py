"""Tests for trigger patterns and importance scoring."""

import pytest

from asuman_memory.triggers import (
    should_trigger,
    score_importance,
    get_confidence_tier,
)


class TestShouldTrigger:
    # Turkish triggers
    def test_hatirla(self):
        assert should_trigger("hatırlıyor musun") is True

    def test_hatirla_ascii(self):
        assert should_trigger("hatirliyor musun") is True

    def test_gecen_hafta(self):
        assert should_trigger("geçen hafta ne konuştuk") is True

    def test_ne_zaman(self):
        assert should_trigger("ne zaman söyledim bunu") is True

    def test_karar(self):
        assert should_trigger("bu konuda karar vermiştik") is True

    def test_unutma(self):
        assert should_trigger("bunu unutma lütfen") is True

    # English triggers
    def test_remember(self):
        assert should_trigger("do you remember what I said") is True

    def test_last_time(self):
        assert should_trigger("last time we talked about this") is True

    def test_my_preference(self):
        assert should_trigger("my favorite color is blue") is True

    def test_question_mark(self):
        assert should_trigger("what was that thing we discussed?") is True

    # Anti-triggers
    def test_greeting(self):
        assert should_trigger("merhaba") is False

    def test_ok(self):
        assert should_trigger("ok") is False
        assert should_trigger("tamam") is False

    def test_too_short(self):
        assert should_trigger("hi") is False

    def test_single_emoji(self):
        assert should_trigger("👍") is False

    def test_generic_word(self):
        assert should_trigger("bu") is False

    # Mixed Turkish+English
    def test_mixed(self):
        assert should_trigger("remember dün ne konuştuk?") is True

    # Past tense
    def test_past_tense_turkish(self):
        assert should_trigger("dün bunu yapmıştık beraber") is True


class TestImportanceScoring:
    def test_question(self):
        score = score_importance("Bu dosyayı nereye koyduk?")
        assert score > 0.5

    def test_importance_markers(self):
        score = score_importance("Bu çok önemli bir karar, unutma!")
        assert score > 0.7

    def test_noise(self):
        score = score_importance("ok")
        assert score < 0.4

    def test_substantive(self):
        long_text = "Bu toplantıda önemli konular tartışıldı. " * 20
        score = score_importance(long_text)
        assert score > 0.5

    def test_user_role_bonus(self):
        base = score_importance("test message")
        user = score_importance("test message", {"role": "user"})
        assert user > base


class TestConfidenceTier:
    def test_high(self):
        assert get_confidence_tier(0.9) == "HIGH"

    def test_medium(self):
        assert get_confidence_tier(0.7) == "MEDIUM"

    def test_low(self):
        assert get_confidence_tier(0.3) == "LOW"

    def test_boundaries(self):
        assert get_confidence_tier(0.85) == "MEDIUM"  # exactly 0.85 is MEDIUM
        assert get_confidence_tier(0.86) == "HIGH"
        assert get_confidence_tier(0.60) == "MEDIUM"
        assert get_confidence_tier(0.59) == "LOW"
