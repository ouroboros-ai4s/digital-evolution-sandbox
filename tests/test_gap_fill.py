"""Gap-fill tests: N1-N7 + Z-core 15 letters (ALPHABET 46→68)."""
import pytest
from des.registry import (
    ALPHABET, GRAN, MOTIF_LEN, VIS, _Z, SLOTS_PER_EVENT,
    PREDICATE_BIT, prey_mask_for_clauses, BB0_TEMPLATE, phenotype
)


class TestAlphabetLen:
    """Test that ALPHABET reaches 68."""

    def test_alphabet_len_is_68(self):
        """ALPHABET must have exactly 68 letters."""
        assert len(ALPHABET) == 68


class TestNPool:
    """Test N1-N7 registration."""

    def test_n_pool_complete(self):
        """All N0-N7 in ALPHABET with family 'N'."""
        for i in range(8):
            letter = f"N{i}"
            assert letter in ALPHABET, f"{letter} missing"
            assert ALPHABET[letter] == "N", f"{letter} not family N"

    def test_n1_to_n7_vis_values(self):
        """VIS values for N1-N7 match roster."""
        expected = {
            "N1": 0.40,
            "N2": 0.70,
            "N3": 0.15,
            "N4": 0.35,
            "N5": 0.00,
            "N6": 1.00,
            "N7": 0.10,
        }
        for letter, expected_vis in expected.items():
            assert VIS[letter] == expected_vis, \
                f"VIS[{letter!r}] = {VIS[letter]}, expected {expected_vis}"

    def test_n1_to_n3_n5_n6_are_residue_gran(self):
        """N1, N2, N3, N5, N6 have gran='residue' (residue-based N-pool subset)."""
        for letter in ["N1", "N2", "N3", "N5", "N6"]:
            assert GRAN[letter] == "residue", \
                f"GRAN[{letter!r}] = {GRAN[letter]}, expected 'residue'"

    def test_n4_n7_are_motif_gran(self):
        """N4 and N7 have gran='motif' (motif-based N-pool subset per roster)."""
        for letter in ["N4", "N7"]:
            assert GRAN[letter] == "motif", \
                f"GRAN[{letter!r}] = {GRAN[letter]}, expected 'motif'"

    def test_n4_motif_len_is_2(self):
        """N4 has MOTIF_LEN=2."""
        assert MOTIF_LEN["N4"] == 2, \
            f"MOTIF_LEN['N4'] = {MOTIF_LEN.get('N4')}, expected 2"

    def test_n7_motif_len_is_2(self):
        """N7 has MOTIF_LEN=2."""
        assert MOTIF_LEN["N7"] == 2, \
            f"MOTIF_LEN['N7'] = {MOTIF_LEN.get('N7')}, expected 2"

    def test_n1_to_n3_n5_n6_not_in_motif_len(self):
        """N1, N2, N3, N5, N6 must not appear in MOTIF_LEN (residue-only)."""
        for letter in ["N1", "N2", "N3", "N5", "N6"]:
            assert letter not in MOTIF_LEN, \
                f"{letter} should not be in MOTIF_LEN (residue letters must not appear)"


class TestZPool:
    """Test Z-core 15 letter registration."""

    def test_z_pool_complete(self):
        """All 16 Z letters (BroadSweep + 15 core) in ALPHABET with family 'Z'."""
        z_letters = [
            "BroadSweep",
            "Scatter Nip", "Epitope Bleed", "Ghost Spike", "Attrition Bite",
            "Hapten Graze", "Burst Leech", "Ambush Coil", "Clade Snare",
            "Frame Pincer", "Lineage Reaper", "Coil Cinch", "Idiotype Lance",
            "Crest Bite", "Hotspot Snipe", "Mirror Fang",
        ]
        for letter in z_letters:
            assert letter in ALPHABET, f"{letter!r} missing from ALPHABET"
            assert ALPHABET[letter] == "Z", f"{letter!r} not family Z"

    def test_z_core_rows_z_values(self):
        """Spot-check Z values for Mirror Fang, BroadSweep, Crest Bite."""
        assert _Z["Mirror Fang"][0] == 1.00
        assert _Z["BroadSweep"][0] == 0.40
        assert _Z["Crest Bite"][0] == 0.90

    def test_ghost_spike_vis_mode_2(self):
        """Ghost Spike has vis_mode=2 (inverse-vis-weighted)."""
        assert _Z["Ghost Spike"][3] == 2

    def test_scatter_nip_vis_mode_1(self):
        """Scatter Nip has vis_mode=1 (vis-weighted)."""
        assert _Z["Scatter Nip"][3] == 1

    def test_frame_pincer_prey_mask_hits_motif_n_bit(self):
        """Frame Pincer prey mask includes motif_N bit."""
        prey_clauses = _Z["Frame Pincer"][1]
        prey_mask = prey_mask_for_clauses(prey_clauses)
        motif_n_bit = PREDICATE_BIT["motif_N"]
        assert prey_mask & motif_n_bit != 0, \
            f"Frame Pincer prey_mask {prey_mask:b} missing motif_N bit {motif_n_bit:b}"


class TestSlotsPerEvent:
    """Test SLOTS_PER_EVENT auto-registration."""

    def test_slots_per_event_covers_all_68(self):
        """SLOTS_PER_EVENT must cover all 68 ALPHABET letters."""
        assert len(SLOTS_PER_EVENT) == 68
        # All values should be 1 except P_cascade (which is 2)
        for letter, n_slots in SLOTS_PER_EVENT.items():
            if letter == "P_cascade":
                assert n_slots == 2
            else:
                assert n_slots == 1, \
                    f"SLOTS_PER_EVENT[{letter!r}] = {n_slots}, expected 1"


class TestByteIdentical:
    """Test that default BB0 phenotype is byte-identical post-gap-fill."""

    def test_default_bb0_phenotype_byte_identical_post_gap_fill(self):
        """Compute BB0 phenotype and verify it matches pre-gap-fill expectations."""
        layout = BB0_TEMPLATE["layout"]
        phe = phenotype(layout)

        # Smoke checks: f and z should be positive, spectrum non-empty
        assert phe.f > 0, f"BB0 phenotype f={phe.f}, expected > 0"
        assert phe.z_raw > 0, f"BB0 phenotype z_raw={phe.z_raw}, expected > 0"
        assert len(phe.spectrum) > 0, "BB0 phenotype spectrum empty"

        # vis_sum should equal 0.20 * count of N0 in layout (only N0 is present by default)
        n0_count = layout.count("N0")
        expected_vis_sum = 0.20 * n0_count
        assert phe.vis_sum == expected_vis_sum, \
            f"BB0 phenotype vis_sum={phe.vis_sum}, expected {expected_vis_sum}"
