"""MiroFish adapter: bridge MiroFish simulation data into the CFT framework.

Reads agent profiles and interaction data from a MiroFish simulation directory,
computes affinity matrices from social interactions, and runs theory predictions.

Requires optional dependencies: pip install cft[mirofish]
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np

from ..theories.base import Agent, Group, TheoryParameters, BehaviorTheory

logger = logging.getLogger(__name__)

# MBTI dimension mappings: each letter → numeric value on a [-1, 1] scale
_MBTI_MAP = {
    "E": 1.0, "I": -1.0,  # Extraversion / Introversion
    "S": 1.0, "N": -1.0,  # Sensing / Intuition
    "T": 1.0, "F": -1.0,  # Thinking / Feeling
    "J": 1.0, "P": -1.0,  # Judging / Perceiving
}

# Default interaction weights for affinity computation
DEFAULT_WEIGHTS: Dict[str, float] = {
    "follow": 0.3,
    "like": 0.2,
    "repost": 0.4,
    "pos_comment": 0.3,
    "neg_comment": -0.3,
}


def _import_pandas():
    """Lazy import of pandas (optional dependency)."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for MiroFish integration. "
            "Install with: pip install cft[mirofish]"
        )


def _import_networkx():
    """Lazy import of networkx (optional dependency)."""
    try:
        import networkx as nx
        return nx
    except ImportError:
        raise ImportError(
            "networkx is required for community detection. "
            "Install with: pip install cft[mirofish]"
        )


def _import_csv():
    """Standard-library csv is always available; just return it."""
    import csv
    return csv


def mbti_to_features(mbti: str) -> np.ndarray:
    """Convert 4-letter MBTI type to 4 continuous features in [-1, 1].

    Parameters
    ----------
    mbti : str
        Four-letter MBTI type (e.g., "ENFP", "ISTJ").

    Returns
    -------
    np.ndarray
        Shape (4,) with values in {-1.0, 1.0}.
    """
    mbti = mbti.upper().strip()
    if len(mbti) != 4:
        raise ValueError(f"MBTI type must be 4 characters, got '{mbti}'")

    features = []
    for i, char in enumerate(mbti):
        if char not in _MBTI_MAP:
            raise ValueError(f"Invalid MBTI character '{char}' at position {i}")
        features.append(_MBTI_MAP[char])
    return np.array(features, dtype=float)


class MiroFishAdapter:
    """Adapter to load MiroFish simulation data into the CFT framework.

    MiroFish produces agent profiles (with MBTI types, opinions, influence scores)
    and interaction logs (follows, likes, reposts, comments). This adapter:

    1. Parses these files into Agent objects with numeric feature vectors
    2. Computes affinity matrices from weighted social interactions
    3. Detects ground-truth communities via Louvain on the interaction graph
    4. Runs a full prediction pipeline: load → predict → score

    Parameters
    ----------
    simulation_dir : str or Path
        Directory containing MiroFish output files.
    """

    def __init__(self, simulation_dir: Union[str, Path]):
        self.simulation_dir = Path(simulation_dir)
        if not self.simulation_dir.is_dir():
            raise FileNotFoundError(
                f"Simulation directory not found: {self.simulation_dir}"
            )
        self._agents: Optional[List[Agent]] = None
        self._interactions = None  # DataFrame, loaded lazily

    def load_agents(self, profiles_file: str = "profiles.jsonl") -> List[Agent]:
        """Load agent profiles from JSONL file.

        Each line should be a JSON object with at least an "id" field.
        Optional fields: "mbti", "opinions" (list of floats), "influence" (float).

        Features are constructed as: [mbti_features..., opinions..., influence].

        Parameters
        ----------
        profiles_file : str
            Filename within simulation_dir (default: "profiles.jsonl").

        Returns
        -------
        List[Agent]
            Agents with numeric feature vectors and metadata.
        """
        path = self.simulation_dir / profiles_file
        if not path.exists():
            raise FileNotFoundError(f"Profiles file not found: {path}")

        agents = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line %d in %s: %s", line_num, path, e)
                    continue

                if "id" not in record:
                    logger.warning("Skipping line %d: missing 'id' field", line_num)
                    continue

                agent_id = int(record["id"])
                feature_parts = []
                metadata: Dict[str, Any] = {}

                # MBTI → 4 features
                if "mbti" in record:
                    try:
                        mbti_feats = mbti_to_features(record["mbti"])
                        feature_parts.append(mbti_feats)
                        metadata["mbti"] = record["mbti"].upper()
                    except ValueError as e:
                        logger.warning("Agent %d: invalid MBTI '%s': %s", agent_id, record["mbti"], e)

                # Opinion vector
                if "opinions" in record:
                    opinions = np.array(record["opinions"], dtype=float)
                    feature_parts.append(opinions)
                    metadata["opinions"] = record["opinions"]

                # Influence score
                if "influence" in record:
                    influence = float(record["influence"])
                    feature_parts.append(np.array([influence]))
                    metadata["influence"] = influence

                if not feature_parts:
                    logger.warning("Agent %d: no feature data found, using zero vector", agent_id)
                    feature_parts.append(np.zeros(1))

                features = np.concatenate(feature_parts)
                agents.append(Agent(id=agent_id, features=features, metadata=metadata))

        if not agents:
            raise ValueError(f"No valid agents found in {path}")

        self._agents = agents
        logger.info("Loaded %d agents from %s", len(agents), path)
        return agents

    def load_interactions(
        self,
        source: str = "jsonl",
        filename: Optional[str] = None,
        before: Optional[str] = None,
    ):
        """Load interaction data from JSONL or SQLite.

        Parameters
        ----------
        source : str
            "jsonl" or "sqlite".
        filename : str, optional
            Override default filename (actions.jsonl or interactions.db).
        before : str, optional
            ISO timestamp cutoff - only include interactions before this time.

        Returns
        -------
        pandas.DataFrame
            Columns: timestamp, agent_i, agent_j, action.
        """
        pd = _import_pandas()

        if source == "jsonl":
            path = self.simulation_dir / (filename or "actions.jsonl")
            if not path.exists():
                raise FileNotFoundError(f"Interactions file not found: {path}")
            records = []
            with open(path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed line %d in %s", line_num, path)
                        continue
                    records.append(record)
            df = pd.DataFrame(records)

        elif source == "sqlite":
            import sqlite3
            path = self.simulation_dir / (filename or "interactions.db")
            if not path.exists():
                raise FileNotFoundError(f"Database not found: {path}")
            conn = sqlite3.connect(str(path))
            df = pd.read_sql("SELECT * FROM interactions", conn)
            conn.close()

        else:
            raise ValueError(f"Unknown source: {source!r}. Use 'jsonl' or 'sqlite'.")

        # Validate required columns
        required = {"agent_i", "agent_j", "action"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Interaction data missing required columns: {missing}")

        # Temporal filtering
        if before is not None and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"] < pd.to_datetime(before)]

        self._interactions = df
        logger.info("Loaded %d interactions from %s", len(df), path)
        return df

    def compute_affinity_matrix(
        self,
        interactions=None,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute signed affinity matrix from interaction data.

        Aggregates weighted interactions between agent pairs, symmetrizes,
        and normalizes to [-1, 1].

        Parameters
        ----------
        interactions : DataFrame, optional
            If None, uses previously loaded interactions.
        weights : dict, optional
            Action type → weight mapping. Defaults to DEFAULT_WEIGHTS.

        Returns
        -------
        np.ndarray
            Symmetric (n_agents, n_agents) affinity matrix in [-1, 1].
        """
        pd = _import_pandas()

        if interactions is None:
            interactions = self._interactions
        if interactions is None:
            raise ValueError("No interactions loaded. Call load_interactions() first.")
        if self._agents is None:
            raise ValueError("No agents loaded. Call load_agents() first.")

        weights = weights or DEFAULT_WEIGHTS
        n = len(self._agents)

        # Map agent IDs to consecutive indices
        id_to_idx = {a.id: idx for idx, a in enumerate(self._agents)}

        affinity = np.zeros((n, n), dtype=float)

        for _, row in interactions.iterrows():
            i = id_to_idx.get(int(row["agent_i"]))
            j = id_to_idx.get(int(row["agent_j"]))
            if i is None or j is None:
                continue  # Skip interactions with unknown agents

            action = row["action"]
            w = weights.get(action, 0.0)
            if w == 0.0 and action not in weights:
                logger.warning("Unknown action type: %s", action)
            affinity[i, j] += w
            affinity[j, i] += w  # Symmetrize

        # Normalize to [-1, 1]
        max_abs = np.max(np.abs(affinity))
        if max_abs > 0:
            affinity /= max_abs

        # Self-affinity = 1.0
        np.fill_diagonal(affinity, 1.0)

        return affinity

    def extract_ground_truth_groups(
        self,
        interactions=None,
        weights: Optional[Dict[str, float]] = None,
        resolution: float = 1.0,
    ) -> List[Group]:
        """Detect ground-truth communities via Louvain on the interaction graph.

        Parameters
        ----------
        interactions : DataFrame, optional
            If None, uses previously loaded interactions.
        weights : dict, optional
            Action type → weight mapping for edge weights.
        resolution : float
            Louvain resolution parameter (higher = more groups).

        Returns
        -------
        List[Group]
            Detected communities as Group objects.
        """
        nx = _import_networkx()
        pd = _import_pandas()

        if interactions is None:
            interactions = self._interactions
        if interactions is None:
            raise ValueError("No interactions loaded. Call load_interactions() first.")
        if self._agents is None:
            raise ValueError("No agents loaded. Call load_agents() first.")

        weights = weights or DEFAULT_WEIGHTS

        # Build weighted graph
        G = nx.Graph()
        for a in self._agents:
            G.add_node(a.id)

        for _, row in interactions.iterrows():
            i, j = int(row["agent_i"]), int(row["agent_j"])
            action = row["action"]
            w = weights.get(action, 0.0)
            if w <= 0:
                continue  # Skip negative/zero interactions for community detection
            if G.has_edge(i, j):
                G[i][j]["weight"] += w
            else:
                G.add_edge(i, j, weight=w)

        # Louvain community detection
        communities = nx.community.louvain_communities(G, resolution=resolution, seed=42)

        groups = []
        for gid, community in enumerate(communities):
            groups.append(Group(id=gid, members=sorted(community)))

        return groups

    def prediction_pipeline(
        self,
        theory_configs: Dict[str, Dict[str, Any]],
        t_freeze: Optional[str] = None,
        t_max: float = 10.0,
        dt: float = 1.0,
    ) -> Dict[str, Any]:
        """End-to-end pipeline: load → predict → score.

        Parameters
        ----------
        theory_configs : dict
            Mapping of theory name → dict with "class" (BehaviorTheory subclass)
            and any keyword arguments for the constructor.
            Example: {"CFT": {"class": CFT, "threshold": 0.5}}
        t_freeze : str, optional
            ISO timestamp to split interactions. Interactions before this time
            are used for affinity; communities after are ground truth.
        t_max : float
            Simulation duration.
        dt : float
            Time step.

        Returns
        -------
        dict
            Keys: "agents", "affinity_matrix", "ground_truth", "histories",
                  "scores", "rankings".
        """
        from ..tournament import PredictionTournament

        if self._agents is None:
            raise ValueError("No agents loaded. Call load_agents() first.")
        if self._interactions is None:
            raise ValueError("No interactions loaded. Call load_interactions() first.")

        pd = _import_pandas()

        # Split interactions temporally if t_freeze is provided
        if t_freeze is not None and "timestamp" in self._interactions.columns:
            ts = pd.to_datetime(self._interactions["timestamp"])
            freeze_ts = pd.to_datetime(t_freeze)
            train_interactions = self._interactions[ts < freeze_ts]
            eval_interactions = self._interactions[ts >= freeze_ts]
        else:
            train_interactions = self._interactions
            eval_interactions = self._interactions

        # Compute affinity from training interactions
        affinity = self.compute_affinity_matrix(train_interactions)

        # Ground truth from evaluation interactions
        ground_truth = self.extract_ground_truth_groups(eval_interactions)

        # Build tournament
        n = len(self._agents)
        n_features = len(self._agents[0].features)
        params = TheoryParameters(n_agents=n, n_features=n_features, random_seed=42)

        tournament = PredictionTournament(self._agents, params)

        for name, config in theory_configs.items():
            theory_cls = config.pop("class")
            tournament.add_theory(name, theory_cls, affinity_matrix=affinity, **config)

        histories = tournament.run(t_max=t_max, dt=dt)
        scores = tournament.score(ground_truth)
        rankings = tournament.rankings(ground_truth)

        return {
            "agents": self._agents,
            "affinity_matrix": affinity,
            "ground_truth": ground_truth,
            "histories": histories,
            "scores": scores,
            "rankings": rankings,
        }

    # ── OASIS Format Support (Issue #26) ──────────────────────────────────────

    @classmethod
    def from_oasis_dir(
        cls,
        sim_dir: Union[str, Path],
        platform: str = "auto",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "MiroFishAdapter":
        """Create a MiroFishAdapter from a MiroFish-Offline / OASIS simulation directory.

        OASIS simulations produce CSV agent profiles and JSONL event logs with
        ``action_type`` / ``action_args`` fields. This constructor converts them
        into the normalized ``profiles.jsonl`` + ``actions.jsonl`` format that
        MiroFishAdapter expects, then loads and returns a ready-to-use adapter.

        Parameters
        ----------
        sim_dir : str or Path
            OASIS simulation directory (e.g. ``simulations/sim_XXXXX/``).
        platform : str
            Platform hint for parsing: ``"reddit"``, ``"twitter"``, or
            ``"auto"`` (default). Currently informational only.
        output_dir : str or Path, optional
            Where to write normalized files. If None, a temporary directory is
            created and tracked for ``cleanup_oasis()``.

        Returns
        -------
        MiroFishAdapter
            Fully initialized adapter with agents and interactions loaded.
        """
        import tempfile

        sim_dir = Path(sim_dir)
        if not sim_dir.is_dir():
            raise FileNotFoundError(
                f"OASIS simulation directory not found: {sim_dir}"
            )

        profile_path = cls._find_oasis_profile(sim_dir)
        events_path = cls._find_oasis_events(sim_dir)

        profiles = cls._parse_oasis_profiles(profile_path)
        actions = cls._parse_oasis_events(events_path, profiles)

        if output_dir is None:
            out_dir = Path(tempfile.mkdtemp(prefix="cft_oasis_"))
            created_temp = True
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            created_temp = False

        profiles_out = out_dir / "profiles.jsonl"
        actions_out = out_dir / "actions.jsonl"

        with open(profiles_out, "w", encoding="utf-8") as f:
            for p in profiles:
                f.write(json.dumps(p) + "\n")

        with open(actions_out, "w", encoding="utf-8") as f:
            for a in actions:
                f.write(json.dumps(a) + "\n")

        adapter = cls(out_dir)
        adapter._oasis_temp_dir = out_dir if created_temp else None
        adapter.load_agents()
        adapter.load_interactions()
        return adapter

    def cleanup_oasis(self) -> None:
        """Remove the temporary directory created by :meth:`from_oasis_dir`, if any."""
        import shutil

        tmp = getattr(self, "_oasis_temp_dir", None)
        if tmp is not None and Path(tmp).exists():
            shutil.rmtree(tmp, ignore_errors=True)
            self._oasis_temp_dir = None

    @staticmethod
    def _find_oasis_profile(sim_dir: Path) -> Path:
        """Locate agent profile CSV in an OASIS simulation directory."""
        candidates = list(sim_dir.glob("*.csv")) + list(sim_dir.glob("**/*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV profile file found in {sim_dir}"
            )
        for name in ("agent_profiles", "profiles", "users", "agents"):
            for c in candidates:
                if name in c.stem.lower():
                    return c
        return candidates[0]

    @staticmethod
    def _find_oasis_events(sim_dir: Path) -> Path:
        """Locate event log JSONL in an OASIS simulation directory."""
        candidates = list(sim_dir.glob("*.jsonl")) + list(sim_dir.glob("**/*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No JSONL event file found in {sim_dir}"
            )
        for name in ("events", "actions", "interactions", "log"):
            for c in candidates:
                if name in c.stem.lower():
                    return c
        return candidates[0]

    @staticmethod
    def _parse_oasis_profiles(profile_path: Path) -> List[Dict[str, Any]]:
        """Parse OASIS agent profile CSV into normalized profile dicts."""
        import csv

        profiles = []
        with open(profile_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile: Dict[str, Any] = {}

                # id
                for id_field in ("user_id", "id", "agent_id"):
                    if id_field in row:
                        try:
                            profile["id"] = int(row[id_field])
                        except (ValueError, TypeError):
                            pass
                        break
                if "id" not in profile:
                    continue

                # name (used later for agent_j lookup in events)
                for name_field in ("name", "username", "user_name", "display_name"):
                    if name_field in row and row[name_field]:
                        profile["name"] = str(row[name_field]).strip()
                        break

                # MBTI
                for mbti_field in ("mbti", "personality", "mbti_type"):
                    if mbti_field in row and row[mbti_field]:
                        profile["mbti"] = str(row[mbti_field]).strip().upper()
                        break

                # Numeric opinion columns
                opinions = []
                for col in row:
                    if "opinion" in col.lower() or "stance" in col.lower():
                        try:
                            opinions.append(float(row[col]))
                        except (ValueError, TypeError):
                            pass
                if opinions:
                    profile["opinions"] = opinions

                # Influence
                for inf_field in ("influence", "followers", "influence_score"):
                    if inf_field in row:
                        try:
                            profile["influence"] = float(row[inf_field])
                        except (ValueError, TypeError):
                            pass
                        break

                profiles.append(profile)

        if not profiles:
            raise ValueError(f"No valid profiles found in {profile_path}")
        return profiles

    @staticmethod
    def _parse_oasis_events(
        events_path: Path,
        profiles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse OASIS event log JSONL into normalized action records."""
        # name → id lookup (OASIS often uses names in action_args)
        name_to_id: Dict[str, int] = {}
        for p in profiles:
            if "name" in p:
                name_to_id[str(p["name"])] = int(p["id"])
        id_set = {int(p["id"]) for p in profiles}

        ACTION_MAP: Dict[str, str] = {
            "QUOTE_POST": "repost",
            "REPOST": "repost",
            "LIKE": "like",
            "FOLLOW": "follow",
            "CREATE_COMMENT": "pos_comment",
            "COMMENT": "pos_comment",
            "DISLIKE": "neg_comment",
            "UNLIKE": "neg_comment",
        }

        actions: List[Dict[str, Any]] = []
        with open(events_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed event line %d in %s", line_num, events_path)
                    continue

                action_type = str(record.get("action_type", "")).upper()
                if action_type not in ACTION_MAP:
                    continue
                normalized = ACTION_MAP[action_type]

                # Source agent
                agent_i = None
                for src_field in ("agent_id", "user_id", "source_id", "actor_id"):
                    if src_field in record:
                        try:
                            agent_i = int(record[src_field])
                        except (ValueError, TypeError):
                            pass
                        break

                # Target agent from action_args
                agent_j = None
                args = record.get("action_args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                if action_type in ("QUOTE_POST", "REPOST"):
                    name = args.get("original_author_name") or args.get("author_name")
                    if name and name in name_to_id:
                        agent_j = name_to_id[name]
                elif action_type == "LIKE":
                    name = args.get("author_name") or args.get("post_author")
                    if name and name in name_to_id:
                        agent_j = name_to_id[name]
                elif action_type == "FOLLOW":
                    raw = args.get("target_user_id") or args.get("target_id") or args.get("followed_user_id")
                    if raw is not None:
                        try:
                            agent_j = int(raw)
                        except (ValueError, TypeError):
                            pass
                    if agent_j is None:
                        name = args.get("target_name") or args.get("username")
                        if name and name in name_to_id:
                            agent_j = name_to_id[name]
                elif action_type in ("CREATE_COMMENT", "COMMENT"):
                    name = args.get("post_author") or args.get("author_name") or args.get("target_user")
                    if name and name in name_to_id:
                        agent_j = name_to_id[name]

                # Fallback: direct target fields on the record
                if agent_j is None:
                    for tgt_field in ("target_id", "target_user_id", "recipient_id"):
                        if tgt_field in record:
                            try:
                                agent_j = int(record[tgt_field])
                            except (ValueError, TypeError):
                                pass
                            break

                if agent_i is None or agent_j is None:
                    continue
                if agent_i not in id_set or agent_j not in id_set:
                    continue
                if agent_i == agent_j:
                    continue

                timestamp = (
                    record.get("timestamp")
                    or record.get("created_at")
                    or record.get("time")
                    or "2024-01-01T00:00:00"
                )

                actions.append({
                    "timestamp": str(timestamp),
                    "agent_i": agent_i,
                    "agent_j": agent_j,
                    "action": normalized,
                })

        return actions
