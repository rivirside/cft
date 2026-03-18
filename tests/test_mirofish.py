"""Tests for MiroFish adapter."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cft import Agent, Group
from cft.integrations.mirofish import MiroFishAdapter, mbti_to_features, DEFAULT_WEIGHTS

# Fixtures directory with sample MiroFish data
FIXTURES = Path(__file__).parent / "fixtures"


# --- Skip if optional deps missing ---
pd = pytest.importorskip("pandas")
nx = pytest.importorskip("networkx")


class TestMBTIConversion:
    def test_enfp(self):
        f = mbti_to_features("ENFP")
        assert f.shape == (4,)
        np.testing.assert_array_equal(f, [1.0, -1.0, -1.0, -1.0])

    def test_istj(self):
        f = mbti_to_features("ISTJ")
        np.testing.assert_array_equal(f, [-1.0, 1.0, 1.0, 1.0])

    def test_case_insensitive(self):
        f = mbti_to_features("enfp")
        np.testing.assert_array_equal(f, [1.0, -1.0, -1.0, -1.0])

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="4 characters"):
            mbti_to_features("ENF")

    def test_invalid_character(self):
        with pytest.raises(ValueError, match="Invalid MBTI character"):
            mbti_to_features("XNFP")


class TestLoadAgents:
    def test_load_from_fixtures(self):
        adapter = MiroFishAdapter(FIXTURES)
        agents = adapter.load_agents()
        assert len(agents) == 8
        assert all(isinstance(a, Agent) for a in agents)

    def test_feature_dimensions(self):
        adapter = MiroFishAdapter(FIXTURES)
        agents = adapter.load_agents()
        # 4 MBTI + 3 opinions + 1 influence = 8 features
        assert agents[0].features.shape == (8,)

    def test_mbti_in_metadata(self):
        adapter = MiroFishAdapter(FIXTURES)
        agents = adapter.load_agents()
        assert agents[0].metadata["mbti"] == "ENFP"
        assert agents[3].metadata["mbti"] == "ISTJ"

    def test_influence_in_metadata(self):
        adapter = MiroFishAdapter(FIXTURES)
        agents = adapter.load_agents()
        assert agents[0].metadata["influence"] == 0.9

    def test_missing_file_raises(self):
        adapter = MiroFishAdapter(FIXTURES)
        with pytest.raises(FileNotFoundError):
            adapter.load_agents("nonexistent.jsonl")

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            MiroFishAdapter("/nonexistent/dir")

    def test_minimal_agent(self):
        """Agent with only id gets a zero-vector feature."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.jsonl"
            path.write_text('{"id": 0}\n')
            adapter = MiroFishAdapter(tmp)
            agents = adapter.load_agents()
            assert len(agents) == 1
            assert agents[0].features.shape == (1,)
            assert agents[0].features[0] == 0.0

    def test_skips_malformed_lines(self):
        """Malformed JSON lines are skipped with a warning."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.jsonl"
            path.write_text('{"id": 0, "mbti": "ENFP"}\nNOT JSON\n{"id": 1, "mbti": "ISTJ"}\n')
            adapter = MiroFishAdapter(tmp)
            agents = adapter.load_agents()
            assert len(agents) == 2


class TestLoadInteractions:
    def test_load_jsonl(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        df = adapter.load_interactions(source="jsonl")
        assert len(df) == 20
        assert "agent_i" in df.columns
        assert "agent_j" in df.columns
        assert "action" in df.columns

    def test_temporal_filter(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        df = adapter.load_interactions(source="jsonl", before="2024-01-02T00:00:00")
        # Only Jan 1 interactions (first 8 lines)
        assert len(df) == 8

    def test_invalid_source_raises(self):
        adapter = MiroFishAdapter(FIXTURES)
        with pytest.raises(ValueError, match="Unknown source"):
            adapter.load_interactions(source="csv")

    def test_missing_file_raises(self):
        adapter = MiroFishAdapter(FIXTURES)
        with pytest.raises(FileNotFoundError):
            adapter.load_interactions(filename="nonexistent.jsonl")


class TestComputeAffinity:
    def test_shape_and_symmetry(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        aff = adapter.compute_affinity_matrix()
        assert aff.shape == (8, 8)
        np.testing.assert_array_almost_equal(aff, aff.T)

    def test_diagonal_is_one(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        aff = adapter.compute_affinity_matrix()
        np.testing.assert_array_equal(np.diag(aff), np.ones(8))

    def test_range(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        aff = adapter.compute_affinity_matrix()
        # Off-diagonal should be in [-1, 1]
        mask = ~np.eye(8, dtype=bool)
        assert np.all(aff[mask] >= -1.0)
        assert np.all(aff[mask] <= 1.0)

    def test_positive_interactions_positive_affinity(self):
        """Agents 0 and 1 have mutual follows → positive affinity."""
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        aff = adapter.compute_affinity_matrix()
        assert aff[0, 1] > 0

    def test_negative_interactions_negative_affinity(self):
        """Agents 0 and 3 have mutual neg_comments → negative affinity."""
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        aff = adapter.compute_affinity_matrix()
        assert aff[0, 3] < 0

    def test_custom_weights(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        # Make follows worth nothing, likes worth everything
        custom = {"follow": 0.0, "like": 1.0, "repost": 0.0, "pos_comment": 0.0, "neg_comment": 0.0}
        aff = adapter.compute_affinity_matrix(weights=custom)
        # Agent 0 & 1 only had follow interactions → should be 0
        assert aff[0, 1] == 0.0

    def test_no_agents_raises(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_interactions = lambda: None  # bypass
        adapter._interactions = pd.DataFrame({"agent_i": [0], "agent_j": [1], "action": ["like"]})
        with pytest.raises(ValueError, match="No agents loaded"):
            adapter.compute_affinity_matrix()


class TestExtractGroundTruth:
    def test_returns_groups(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        groups = adapter.extract_ground_truth_groups()
        assert len(groups) > 0
        assert all(isinstance(g, Group) for g in groups)

    def test_all_agents_assigned(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        groups = adapter.extract_ground_truth_groups()
        all_members = set()
        for g in groups:
            all_members.update(g.members)
        # All 8 agents should appear
        assert all_members == {0, 1, 2, 3, 4, 5, 6, 7}

    def test_resolution_affects_group_count(self):
        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()
        groups_low = adapter.extract_ground_truth_groups(resolution=0.1)
        groups_high = adapter.extract_ground_truth_groups(resolution=5.0)
        # Higher resolution should produce more or equal groups
        assert len(groups_high) >= len(groups_low)


class TestPredictionPipeline:
    def test_end_to_end(self):
        from cft import CFT

        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()

        result = adapter.prediction_pipeline(
            theory_configs={"CFT": {"class": CFT, "threshold": 0.3}},
            t_max=2.0,
            dt=1.0,
        )
        assert "agents" in result
        assert "affinity_matrix" in result
        assert "ground_truth" in result
        assert "histories" in result
        assert "scores" in result
        assert "rankings" in result
        assert "CFT" in result["scores"]

    def test_with_temporal_split(self):
        from cft import CFT

        adapter = MiroFishAdapter(FIXTURES)
        adapter.load_agents()
        adapter.load_interactions()

        result = adapter.prediction_pipeline(
            theory_configs={"CFT": {"class": CFT, "threshold": 0.3}},
            t_freeze="2024-01-02T00:00:00",
            t_max=2.0,
            dt=1.0,
        )
        assert "CFT" in result["scores"]
        assert 0.0 <= result["scores"]["CFT"]["pas"] <= 1.0


# ---------------------------------------------------------------------------
# Issue #26 - from_oasis_dir
# ---------------------------------------------------------------------------


def _write_oasis_dir(tmp_path, n_agents=4, n_events=10):
    """Write minimal OASIS-format CSV + JSONL into tmp_path."""
    import csv

    # profiles CSV
    profile_path = tmp_path / "agent_profiles.csv"
    mbti_types = ["INTJ", "ENFP", "ISTJ", "ENTP"]
    with open(profile_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "name", "mbti", "influence"])
        writer.writeheader()
        for i in range(n_agents):
            writer.writerow({
                "user_id": i,
                "name": f"agent_{i}",
                "mbti": mbti_types[i % len(mbti_types)],
                "influence": round(0.1 * (i + 1), 1),
            })

    # events JSONL
    events_path = tmp_path / "events.jsonl"
    action_types = ["FOLLOW", "LIKE", "REPOST", "CREATE_COMMENT"]
    with open(events_path, "w") as f:
        for k in range(n_events):
            src = k % n_agents
            tgt = (k + 1) % n_agents
            atype = action_types[k % len(action_types)]
            if atype == "FOLLOW":
                args = {"target_user_id": tgt}
            elif atype == "LIKE":
                args = {"author_name": f"agent_{tgt}"}
            elif atype == "REPOST":
                args = {"original_author_name": f"agent_{tgt}"}
            else:
                args = {"post_author": f"agent_{tgt}"}
            record = {
                "action_type": atype,
                "agent_id": src,
                "action_args": args,
                "timestamp": f"2024-01-0{(k % 3) + 1}T00:00:00",
            }
            f.write(json.dumps(record) + "\n")

    return tmp_path


class TestFromOasisDir:
    def test_basic_load(self, tmp_path):
        _write_oasis_dir(tmp_path, n_agents=4, n_events=12)
        adapter = MiroFishAdapter.from_oasis_dir(tmp_path)
        assert len(adapter._agents) == 4

    def test_agents_have_mbti_features(self, tmp_path):
        _write_oasis_dir(tmp_path, n_agents=4, n_events=12)
        adapter = MiroFishAdapter.from_oasis_dir(tmp_path)
        for agent in adapter._agents:
            # 4 MBTI + 1 influence = 5 features
            assert agent.features.shape[0] >= 4

    def test_interactions_loaded(self, tmp_path):
        _write_oasis_dir(tmp_path, n_agents=4, n_events=12)
        adapter = MiroFishAdapter.from_oasis_dir(tmp_path)
        assert adapter._interactions is not None
        assert len(adapter._interactions) > 0

    def test_affinity_matrix_symmetric(self, tmp_path):
        _write_oasis_dir(tmp_path, n_agents=4, n_events=16)
        adapter = MiroFishAdapter.from_oasis_dir(tmp_path)
        aff = adapter.compute_affinity_matrix()
        np.testing.assert_array_almost_equal(aff, aff.T)

    def test_no_self_interactions(self, tmp_path):
        _write_oasis_dir(tmp_path, n_agents=4, n_events=16)
        adapter = MiroFishAdapter.from_oasis_dir(tmp_path)
        df = adapter._interactions
        assert (df["agent_i"] != df["agent_j"]).all()

    def test_cleanup_oasis_removes_temp_dir(self, tmp_path):
        import os
        _write_oasis_dir(tmp_path, n_agents=4, n_events=12)
        adapter = MiroFishAdapter.from_oasis_dir(tmp_path)
        tmp_dir = adapter._oasis_temp_dir
        assert tmp_dir is not None
        assert os.path.isdir(tmp_dir)
        adapter.cleanup_oasis()
        assert not os.path.isdir(tmp_dir)

    def test_cleanup_oasis_safe_to_call_twice(self, tmp_path):
        _write_oasis_dir(tmp_path, n_agents=4, n_events=12)
        adapter = MiroFishAdapter.from_oasis_dir(tmp_path)
        adapter.cleanup_oasis()
        adapter.cleanup_oasis()  # should not raise

    def test_output_dir_option(self, tmp_path):
        src = tmp_path / "oasis_sim"
        src.mkdir()
        out = tmp_path / "normalized"
        _write_oasis_dir(src, n_agents=4, n_events=8)
        adapter = MiroFishAdapter.from_oasis_dir(src, output_dir=out)
        assert (out / "profiles.jsonl").exists()
        assert (out / "actions.jsonl").exists()
        # No temp dir created when output_dir is explicit
        assert adapter._oasis_temp_dir is None

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            MiroFishAdapter.from_oasis_dir("/nonexistent/oasis/sim")

    def test_action_type_mapping(self, tmp_path):
        """All four OASIS action types should map to valid CFT actions."""
        _write_oasis_dir(tmp_path, n_agents=4, n_events=20)
        adapter = MiroFishAdapter.from_oasis_dir(tmp_path)
        valid_actions = {"follow", "like", "repost", "pos_comment", "neg_comment"}
        actions_found = set(adapter._interactions["action"].unique())
        assert actions_found <= valid_actions
