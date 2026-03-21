"""Intervention system for social dynamics simulation.

Apply perturbations mid-simulation and measure how group structures respond.
Supports theory-agnostic interventions (feature shifts, agent removal) and
theory-specific ones (DCT proximity/alignment manipulation).

Usage::

    from cft import DCT, TheoryParameters, Agent
    from cft.interventions import (
        InterventionRunner, RemoveAgents, ShiftFeatures,
        AddAgent, NoiseShock, ModifyAffinity,
    )

    theory = DCT(params, mu=0.3, lam=0.05)
    theory.initialize_agents(agents)

    runner = InterventionRunner(theory, [
        RemoveAgents(time=5.0, agent_ids=[3]),
        ShiftFeatures(time=10.0, agent_ids=[0, 1], delta=np.array([0.5, 0, -0.3])),
    ])
    report = runner.run(t_max=30.0, dt=1.0)
    print(report.resilience_scores)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .theories.base import Agent, BehaviorTheory, Group, TheoryParameters
from .comparator import TheoryComparator


# ---------------------------------------------------------------------------
# Intervention base
# ---------------------------------------------------------------------------

@dataclass
class Intervention:
    """A perturbation applied to a running simulation at a specific time.

    Subclasses implement ``apply()`` to mutate the theory's internal state.
    """

    time: float

    def apply(self, theory: BehaviorTheory) -> Dict[str, Any]:
        """Mutate *theory* in place. Return a log dict describing what changed."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete interventions
# ---------------------------------------------------------------------------

@dataclass
class RemoveAgents(Intervention):
    """Deactivate agents by zeroing their affinities.

    Rather than physically removing agents (which would break index-aligned
    arrays), this sets all affinities involving the target agents to zero,
    making them isolated singletons.
    """

    agent_ids: List[int] = field(default_factory=list)

    def apply(self, theory: BehaviorTheory) -> Dict[str, Any]:
        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        removed = []
        for aid in self.agent_ids:
            if aid not in idx_map:
                continue
            idx = idx_map[aid]
            if theory._affinity_matrix is not None:
                theory._affinity_matrix[idx, :] = 0.0
                theory._affinity_matrix[:, idx] = 0.0
                theory._affinity_matrix[idx, idx] = 1.0
            # DCT-specific: isolate in both layers
            if hasattr(theory, "context_pos") and theory.context_pos is not None:
                theory.context_pos[idx] = 1e6 * np.ones_like(theory.context_pos[idx])
            if hasattr(theory, "_proximity_affinity") and theory._proximity_affinity is not None:
                theory._proximity_affinity[idx, :] = 0.0
                theory._proximity_affinity[:, idx] = 0.0
                theory._proximity_affinity[idx, idx] = 1.0
            if hasattr(theory, "_alignment_affinity") and theory._alignment_affinity is not None:
                theory._alignment_affinity[idx, :] = 0.0
                theory._alignment_affinity[:, idx] = 0.0
                theory._alignment_affinity[idx, idx] = 1.0
            if hasattr(theory, "_effective_affinity") and theory._effective_affinity is not None:
                theory._effective_affinity[idx, :] = 0.0
                theory._effective_affinity[:, idx] = 0.0
                theory._effective_affinity[idx, idx] = 1.0
            removed.append(aid)
        return {"type": "remove_agents", "removed": removed}


@dataclass
class ShiftFeatures(Intervention):
    """Shift agent feature vectors by a delta.

    Models propaganda, opinion change, or economic shock targeting specific
    agents.  After shifting, recalculates affinity if available.
    """

    agent_ids: List[int] = field(default_factory=list)
    delta: Optional[np.ndarray] = None

    def apply(self, theory: BehaviorTheory) -> Dict[str, Any]:
        if self.delta is None:
            return {"type": "shift_features", "shifted": []}

        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        shifted = []
        for aid in self.agent_ids:
            if aid not in idx_map:
                continue
            idx = idx_map[aid]
            agent = theory.agents[idx]
            d = self.delta
            if len(d) < len(agent.features):
                d = np.concatenate([d, np.zeros(len(agent.features) - len(d))])
            elif len(d) > len(agent.features):
                d = d[: len(agent.features)]
            agent.features = agent.features + d
            # DCT: also shift alignment
            if hasattr(theory, "alignment") and theory.alignment is not None:
                theory.alignment[idx] += d[: theory.alignment.shape[1]]
            shifted.append(aid)

        # Recalculate affinities if the theory supports it
        if hasattr(theory, "_update_affinities"):
            theory._update_affinities()
        elif theory._affinity_matrix is not None:
            from .affinity import compute_affinity_matrix
            theory._affinity_matrix = compute_affinity_matrix(theory.agents)

        return {"type": "shift_features", "shifted": shifted}


@dataclass
class AddAgent(Intervention):
    """Inject a new agent into the simulation.

    Grows all internal arrays by one row/column. The new agent starts as a
    singleton and integrates through normal dynamics.
    """

    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply(self, theory: BehaviorTheory) -> Dict[str, Any]:
        if self.features is None:
            return {"type": "add_agent", "added": None}

        new_id = max(a.id for a in theory.agents) + 1
        new_agent = Agent(id=new_id, features=self.features.copy(), metadata=dict(self.metadata))
        theory.agents.append(new_agent)
        n = len(theory.agents)

        # Expand affinity matrix
        if theory._affinity_matrix is not None:
            old = theory._affinity_matrix
            new_aff = np.zeros((n, n))
            new_aff[: n - 1, : n - 1] = old
            # Compute affinities for new agent
            from .affinity import compute_affinity_matrix
            full_aff = compute_affinity_matrix(theory.agents)
            new_aff[n - 1, :] = full_aff[n - 1, :]
            new_aff[:, n - 1] = full_aff[:, n - 1]
            theory._affinity_matrix = new_aff

        # DCT: expand per-agent parameter vectors (mu, lam)
        if hasattr(theory, "mu") and isinstance(getattr(theory, "mu", None), np.ndarray):
            # Default: new agent gets the mean of existing agents' rates
            theory.mu = np.append(theory.mu, np.mean(theory.mu))
        if hasattr(theory, "lam") and isinstance(getattr(theory, "lam", None), np.ndarray):
            theory.lam = np.append(theory.lam, np.mean(theory.lam))

        # DCT: expand context_pos and alignment
        if hasattr(theory, "context_pos") and theory.context_pos is not None:
            d = theory.context_pos.shape[1]
            new_ctx = self.features[:d].copy() if len(self.features) >= d else np.zeros(d)
            theory.context_pos = np.vstack([theory.context_pos, new_ctx.reshape(1, -1)])
        if hasattr(theory, "alignment") and theory.alignment is not None:
            d = theory.alignment.shape[1]
            new_align = self.features[:d].copy() if len(self.features) >= d else np.zeros(d)
            theory.alignment = np.vstack([theory.alignment, new_align.reshape(1, -1)])
        if hasattr(theory, "_update_affinities"):
            # Expand proximity/alignment/effective matrices
            for attr in ("_proximity_affinity", "_alignment_affinity", "_effective_affinity"):
                mat = getattr(theory, attr, None)
                if mat is not None:
                    expanded = np.zeros((n, n))
                    expanded[: n - 1, : n - 1] = mat
                    expanded[n - 1, n - 1] = 1.0
                    setattr(theory, attr, expanded)
            theory._update_affinities()

        # GFT: expand positions
        if hasattr(theory, "positions") and theory.positions is not None:
            d = theory.positions.shape[1]
            new_pos = self.features[:d].copy() if len(self.features) >= d else np.zeros(d)
            theory.positions = np.vstack([theory.positions, new_pos.reshape(1, -1)])

        # TST: expand spins
        if hasattr(theory, "spins") and theory.spins is not None:
            rng = getattr(theory, "_rng", np.random.default_rng(42))
            n_states = getattr(theory, "n_states", 3)
            theory.spins = np.append(theory.spins, rng.integers(0, n_states))

        theory.params.n_agents = n
        return {"type": "add_agent", "added": new_id}


@dataclass
class NoiseShock(Intervention):
    """Apply random noise to agent features, simulating a crisis or disruption.

    If ``target_ids`` is None, all agents are affected.
    """

    intensity: float = 0.5
    target_ids: Optional[List[int]] = None
    seed: Optional[int] = None

    def apply(self, theory: BehaviorTheory) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        targets = self.target_ids if self.target_ids is not None else [a.id for a in theory.agents]
        shocked = []
        for aid in targets:
            if aid not in idx_map:
                continue
            idx = idx_map[aid]
            agent = theory.agents[idx]
            noise = rng.normal(0, self.intensity, size=agent.features.shape)
            agent.features = agent.features + noise
            if hasattr(theory, "alignment") and theory.alignment is not None:
                d = theory.alignment.shape[1]
                theory.alignment[idx] += noise[:d]
            shocked.append(aid)

        if hasattr(theory, "_update_affinities"):
            theory._update_affinities()
        elif theory._affinity_matrix is not None:
            from .affinity import compute_affinity_matrix
            theory._affinity_matrix = compute_affinity_matrix(theory.agents)

        return {"type": "noise_shock", "intensity": self.intensity, "shocked": len(shocked)}


@dataclass
class ModifyAffinity(Intervention):
    """Directly modify pairwise affinities.

    Models platform algorithm changes, communication channel shifts, or
    forced/severed connections.
    """

    pairs: List[Tuple[int, int]] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def apply(self, theory: BehaviorTheory) -> Dict[str, Any]:
        if theory._affinity_matrix is None:
            return {"type": "modify_affinity", "modified": 0}

        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        modified = 0
        for (a, b), val in zip(self.pairs, self.values):
            if a not in idx_map or b not in idx_map:
                continue
            ia, ib = idx_map[a], idx_map[b]
            theory._affinity_matrix[ia, ib] = val
            theory._affinity_matrix[ib, ia] = val
            modified += 1

        return {"type": "modify_affinity", "modified": modified}


@dataclass
class ShiftProximity(Intervention):
    """DCT-specific: move agents in context space.

    Models forced relocation, platform reassignment, or introduction to
    new social circles.
    """

    agent_ids: List[int] = field(default_factory=list)
    delta: Optional[np.ndarray] = None

    def apply(self, theory: BehaviorTheory) -> Dict[str, Any]:
        if self.delta is None or not hasattr(theory, "context_pos") or theory.context_pos is None:
            return {"type": "shift_proximity", "shifted": []}

        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        shifted = []
        for aid in self.agent_ids:
            if aid not in idx_map:
                continue
            idx = idx_map[aid]
            d = theory.context_pos.shape[1]
            delta = self.delta[:d] if len(self.delta) >= d else np.concatenate(
                [self.delta, np.zeros(d - len(self.delta))]
            )
            theory.context_pos[idx] += delta
            shifted.append(aid)

        if hasattr(theory, "_update_affinities"):
            theory._update_affinities()

        return {"type": "shift_proximity", "shifted": shifted}


@dataclass
class ShiftAlignment(Intervention):
    """DCT-specific: directly change agent alignment vectors.

    Models ideological conversion, education, or radicalization.
    """

    agent_ids: List[int] = field(default_factory=list)
    delta: Optional[np.ndarray] = None

    def apply(self, theory: BehaviorTheory) -> Dict[str, Any]:
        if self.delta is None or not hasattr(theory, "alignment") or theory.alignment is None:
            return {"type": "shift_alignment", "shifted": []}

        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        shifted = []
        for aid in self.agent_ids:
            if aid not in idx_map:
                continue
            idx = idx_map[aid]
            d = theory.alignment.shape[1]
            delta = self.delta[:d] if len(self.delta) >= d else np.concatenate(
                [self.delta, np.zeros(d - len(self.delta))]
            )
            theory.alignment[idx] += delta
            shifted.append(aid)

        if hasattr(theory, "_update_affinities"):
            theory._update_affinities()

        return {"type": "shift_alignment", "shifted": shifted}


# ---------------------------------------------------------------------------
# Sustained interventions
# ---------------------------------------------------------------------------

@dataclass
class SustainedIntervention:
    """An intervention applied every step over a time range.

    Unlike point-in-time Intervention, this applies continuously from
    ``start`` to ``end`` (inclusive). Each step within the range calls
    ``apply_step()``.
    """

    start: float
    end: float

    def apply_step(self, theory: BehaviorTheory, current_time: float) -> Dict[str, Any]:
        """Apply one step of the sustained intervention. Return a log dict."""
        raise NotImplementedError

    def is_active(self, current_time: float, dt: float) -> bool:
        """True if this intervention should fire at the current timestep."""
        return self.start <= current_time + dt and current_time < self.end


@dataclass
class SustainedShift(SustainedIntervention):
    """Apply a small feature delta every step over a time range.

    Models sustained propaganda, ongoing economic pressure, or continuous
    platform algorithm bias. The total effect compounds over the duration.
    """

    agent_ids: List[int] = field(default_factory=list)
    delta_per_step: Optional[np.ndarray] = None

    def apply_step(self, theory: BehaviorTheory, current_time: float) -> Dict[str, Any]:
        if self.delta_per_step is None:
            return {"type": "sustained_shift", "shifted": []}

        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        shifted = []
        for aid in self.agent_ids:
            if aid not in idx_map:
                continue
            idx = idx_map[aid]
            agent = theory.agents[idx]
            d = self.delta_per_step
            if len(d) < len(agent.features):
                d = np.concatenate([d, np.zeros(len(agent.features) - len(d))])
            elif len(d) > len(agent.features):
                d = d[: len(agent.features)]
            agent.features = agent.features + d
            if hasattr(theory, "alignment") and theory.alignment is not None:
                theory.alignment[idx] += d[: theory.alignment.shape[1]]
            shifted.append(aid)

        if hasattr(theory, "_update_affinities"):
            theory._update_affinities()
        elif theory._affinity_matrix is not None:
            from .affinity import compute_affinity_matrix
            theory._affinity_matrix = compute_affinity_matrix(theory.agents)

        return {"type": "sustained_shift", "shifted": shifted, "time": current_time}


@dataclass
class SustainedNoise(SustainedIntervention):
    """Apply ongoing random noise every step over a time range.

    Models prolonged instability, crisis conditions, or sustained disruption.
    """

    intensity: float = 0.1
    target_ids: Optional[List[int]] = None
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def apply_step(self, theory: BehaviorTheory, current_time: float) -> Dict[str, Any]:
        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        targets = self.target_ids if self.target_ids is not None else [a.id for a in theory.agents]
        shocked = 0
        for aid in targets:
            if aid not in idx_map:
                continue
            idx = idx_map[aid]
            agent = theory.agents[idx]
            noise = self._rng.normal(0, self.intensity, size=agent.features.shape)
            agent.features = agent.features + noise
            if hasattr(theory, "alignment") and theory.alignment is not None:
                d = theory.alignment.shape[1]
                theory.alignment[idx] += noise[:d]
            shocked += 1

        if hasattr(theory, "_update_affinities"):
            theory._update_affinities()

        return {"type": "sustained_noise", "shocked": shocked, "time": current_time}


@dataclass
class SustainedAffinityBias(SustainedIntervention):
    """Continuously bias specific affinities every step.

    Models platform algorithm changes that persistently promote or suppress
    certain connections.
    """

    pairs: List[Tuple[int, int]] = field(default_factory=list)
    bias_per_step: List[float] = field(default_factory=list)

    def apply_step(self, theory: BehaviorTheory, current_time: float) -> Dict[str, Any]:
        if theory._affinity_matrix is None:
            return {"type": "sustained_affinity_bias", "modified": 0, "time": current_time}

        idx_map = {a.id: i for i, a in enumerate(theory.agents)}
        modified = 0
        for (a, b), bias in zip(self.pairs, self.bias_per_step):
            if a not in idx_map or b not in idx_map:
                continue
            ia, ib = idx_map[a], idx_map[b]
            theory._affinity_matrix[ia, ib] = np.clip(
                theory._affinity_matrix[ia, ib] + bias, -1.0, 1.0
            )
            theory._affinity_matrix[ib, ia] = theory._affinity_matrix[ia, ib]
            modified += 1

        return {"type": "sustained_affinity_bias", "modified": modified, "time": current_time}


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """State of the simulation at a single point in time."""

    time: float
    groups: List[Group]
    n_groups: int
    group_sizes: List[int]
    state: Dict[str, Any]
    intervention_log: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Intervention runner
# ---------------------------------------------------------------------------

class InterventionRunner:
    """Run a theory simulation with interventions injected at specified times.

    Replaces the theory's built-in ``run_simulation`` loop so that
    interventions can be applied between steps. Supports both point-in-time
    :class:`Intervention` and continuous :class:`SustainedIntervention`.
    """

    def __init__(
        self,
        theory: BehaviorTheory,
        interventions: Optional[List[Intervention]] = None,
        sustained: Optional[List[SustainedIntervention]] = None,
    ):
        self.theory = theory
        self.interventions = sorted(interventions or [], key=lambda iv: iv.time)
        self.sustained = sustained or []

    def run(self, t_max: float, dt: float = 1.0) -> "InterventionReport":
        """Run the simulation with interventions, return a report."""
        theory = self.theory
        snapshots: List[Snapshot] = []
        intervention_log: List[Dict[str, Any]] = []
        pending = list(self.interventions)

        # Capture initial state
        snapshots.append(self._snapshot(theory, intervention_log=None))

        while theory.current_time < t_max:
            step_interventions = []

            # Apply point-in-time interventions scheduled for this step
            while pending and pending[0].time <= theory.current_time + dt:
                iv = pending.pop(0)
                log = iv.apply(theory)
                log["time"] = iv.time
                log["intervention_class"] = type(iv).__name__
                intervention_log.append(log)
                step_interventions.append(log)

            # Apply sustained interventions active at this step
            for siv in self.sustained:
                if siv.is_active(theory.current_time, dt):
                    log = siv.apply_step(theory, theory.current_time)
                    log["intervention_class"] = type(siv).__name__
                    intervention_log.append(log)
                    step_interventions.append(log)

            theory.step(dt)

            iv_log = step_interventions if step_interventions else None
            snapshots.append(self._snapshot(theory, intervention_log=iv_log))

        return InterventionReport(snapshots=snapshots, intervention_log=intervention_log)

    @staticmethod
    def _snapshot(theory: BehaviorTheory, intervention_log=None) -> Snapshot:
        groups = theory.get_groups()
        return Snapshot(
            time=theory.current_time,
            groups=groups,
            n_groups=len(groups),
            group_sizes=sorted([len(g.members) for g in groups], reverse=True),
            state=theory.get_state(),
            intervention_log=intervention_log,
        )


# ---------------------------------------------------------------------------
# Report and resilience analysis
# ---------------------------------------------------------------------------

class InterventionReport:
    """Analysis of a simulation run with interventions."""

    def __init__(
        self,
        snapshots: List[Snapshot],
        intervention_log: List[Dict[str, Any]],
    ):
        self.snapshots = snapshots
        self.intervention_log = intervention_log
        self._stability: Optional[List[float]] = None

    @property
    def timeline(self) -> List[Dict[str, Any]]:
        """Compact timeline: time, n_groups, group_sizes, intervention flag."""
        return [
            {
                "time": s.time,
                "n_groups": s.n_groups,
                "group_sizes": s.group_sizes,
                "intervention": s.intervention_log is not None,
            }
            for s in self.snapshots
        ]

    @property
    def stability_curve(self) -> List[float]:
        """NMI between consecutive timesteps. High = stable, drop = fracture."""
        if self._stability is not None:
            return self._stability
        curve = []
        for i in range(1, len(self.snapshots)):
            nmi = TheoryComparator.compare_group_structures(
                self.snapshots[i - 1].groups,
                self.snapshots[i].groups,
                metric="nmi",
            )
            curve.append(nmi)
        self._stability = curve
        return curve

    @property
    def fracture_events(self) -> List[Dict[str, Any]]:
        """Timesteps where group count increased (a group split)."""
        events = []
        for i in range(1, len(self.snapshots)):
            prev, curr = self.snapshots[i - 1], self.snapshots[i]
            if curr.n_groups > prev.n_groups:
                events.append({
                    "time": curr.time,
                    "groups_before": prev.n_groups,
                    "groups_after": curr.n_groups,
                    "nmi": self.stability_curve[i - 1],
                })
        return events

    @property
    def merge_events(self) -> List[Dict[str, Any]]:
        """Timesteps where group count decreased (groups merged)."""
        events = []
        for i in range(1, len(self.snapshots)):
            prev, curr = self.snapshots[i - 1], self.snapshots[i]
            if curr.n_groups < prev.n_groups:
                events.append({
                    "time": curr.time,
                    "groups_before": prev.n_groups,
                    "groups_after": curr.n_groups,
                    "nmi": self.stability_curve[i - 1],
                })
        return events

    def group_survival(self, before_time: float, after_time: float) -> Dict[str, Any]:
        """Which groups from *before_time* still exist at *after_time*?

        Returns survival rate, a list of surviving groups, and a list of
        groups that fractured or dissolved.
        """
        snap_before = self._snap_at(before_time)
        snap_after = self._snap_at(after_time)
        if snap_before is None or snap_after is None:
            return {"error": "time not found in snapshots"}

        before_groups = snap_before.groups
        after_groups = snap_after.groups
        after_sets = [set(g.members) for g in after_groups]

        survived = []
        dissolved = []
        for g in before_groups:
            members = set(g.members)
            # A group "survives" if >50% of its members are still co-grouped
            best_overlap = 0.0
            for ag_set in after_sets:
                overlap = len(members & ag_set) / len(members) if members else 0
                best_overlap = max(best_overlap, overlap)
            if best_overlap > 0.5:
                survived.append({"group_id": g.id, "overlap": best_overlap})
            else:
                dissolved.append({"group_id": g.id, "best_overlap": best_overlap})

        total = len(before_groups)
        return {
            "survival_rate": len(survived) / total if total > 0 else 1.0,
            "survived": survived,
            "dissolved": dissolved,
            "before_groups": total,
            "after_groups": len(after_groups),
        }

    def agent_stability(self) -> Dict[int, float]:
        """Per-agent stability: fraction of timesteps where the agent's
        group membership didn't change."""
        if len(self.snapshots) < 2:
            return {}

        def agent_group(groups, agent_id):
            for g in groups:
                if agent_id in g.members:
                    return g.id
            return -1

        all_ids = set()
        for s in self.snapshots:
            for g in s.groups:
                all_ids.update(g.members)

        stability = {}
        for aid in all_ids:
            stable = 0
            for i in range(1, len(self.snapshots)):
                g_prev = agent_group(self.snapshots[i - 1].groups, aid)
                g_curr = agent_group(self.snapshots[i].groups, aid)
                if g_prev == g_curr:
                    stable += 1
            stability[aid] = stable / (len(self.snapshots) - 1)
        return stability

    def vulnerability_ranking(self) -> List[Tuple[int, float]]:
        """Rank agents by how easily they switch groups (most volatile first)."""
        stab = self.agent_stability()
        return sorted(stab.items(), key=lambda x: x[1])

    @property
    def resilience_scores(self) -> Dict[str, float]:
        """Composite resilience metrics for the entire run."""
        curve = self.stability_curve
        if not curve:
            return {"mean_stability": 1.0, "min_stability": 1.0, "recovery_rate": 1.0}

        # Find intervention timesteps
        iv_indices = []
        for i, s in enumerate(self.snapshots):
            if s.intervention_log is not None:
                iv_indices.append(i)

        # Mean stability across all timesteps
        mean_stab = float(np.mean(curve))

        # Minimum stability (worst disruption)
        min_stab = float(np.min(curve))

        # Recovery rate: after each intervention, how many steps until
        # stability returns above 0.8?
        recovery_steps = []
        for iv_idx in iv_indices:
            curve_idx = iv_idx  # stability_curve[i] = NMI between snapshot[i] and snapshot[i+1]
            if curve_idx >= len(curve):
                continue
            steps = 0
            for j in range(curve_idx, len(curve)):
                if curve[j] >= 0.8:
                    break
                steps += 1
            recovery_steps.append(steps)

        recovery_rate = 1.0
        if recovery_steps:
            max_possible = len(curve)
            avg_recovery = float(np.mean(recovery_steps))
            recovery_rate = max(0.0, 1.0 - avg_recovery / max_possible)

        return {
            "mean_stability": mean_stab,
            "min_stability": min_stab,
            "recovery_rate": recovery_rate,
        }

    def _snap_at(self, time: float) -> Optional[Snapshot]:
        """Find snapshot closest to the given time."""
        best = None
        best_dist = float("inf")
        for s in self.snapshots:
            d = abs(s.time - time)
            if d < best_dist:
                best_dist = d
                best = s
        return best

    def summary(self) -> Dict[str, Any]:
        """Complete summary dict suitable for JSON serialization."""
        return {
            "n_snapshots": len(self.snapshots),
            "n_interventions": len(self.intervention_log),
            "interventions": self.intervention_log,
            "timeline": self.timeline,
            "fracture_events": self.fracture_events,
            "merge_events": self.merge_events,
            "resilience": self.resilience_scores,
            "vulnerability_ranking": [
                {"agent_id": aid, "stability": s}
                for aid, s in self.vulnerability_ranking()
            ],
        }
