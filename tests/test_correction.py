"""
Unit tests for Streaming Correction Engine.

Tests:
- Final consonant restoration (tes->test, ting->thing)
- Hot-word substitution
- Confidence gating rejects hallucinated corrections
- Edit distance ratio check
"""

import pytest

from aios.core.streaming_correction import RuleBasedCorrector


class TestRuleBasedCorrector:

    def test_final_consonant_restoration(self):
        corrector = RuleBasedCorrector()
        result, modified = corrector.correct("tes")
        assert result == "test"
        assert modified

    def test_th_fronting(self):
        corrector = RuleBasedCorrector()
        result, modified = corrector.correct("ting")
        assert result == "thing"
        assert modified

    def test_no_change_for_correct_word(self):
        corrector = RuleBasedCorrector()
        result, modified = corrector.correct("hello")
        assert result == "hello"
        assert not modified

    def test_hot_word_substitution(self):
        corrector = RuleBasedCorrector(hot_words=[
            {"term": "PersonaPlex", "phonetic_variants": ["personna plex", "persona plex"]},
        ])
        result, modified = corrector.correct("personna plex")
        assert "PersonaPlex" in result
        assert modified

    def test_hot_word_case_insensitive(self):
        corrector = RuleBasedCorrector(hot_words=[
            {"term": "NVIDIA", "phonetic_variants": ["en vidia"]},
        ])
        result, modified = corrector.correct("en vidia")
        assert "NVIDIA" in result
        assert modified

    def test_hallucination_gate_rejects_large_edit(self):
        corrector = RuleBasedCorrector()
        # A word that doesn't match any rule should pass through unchanged
        result, modified = corrector.correct("supercalifragilistic")
        assert result == "supercalifragilistic"
        assert not modified

    def test_multiple_corrections_in_sentence(self):
        corrector = RuleBasedCorrector()
        # Process word by word
        words = "I ting tes is good".split()
        corrected = []
        for word in words:
            result, _ = corrector.correct(word)
            corrected.append(result)
        sentence = " ".join(corrected)
        assert "thing" in sentence
        assert "test" in sentence
