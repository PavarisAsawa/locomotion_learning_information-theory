"""Loading and reduction of the rollout dumps behind the paper figures.

Paths resolve from this file rather than the working directory, so the
notebooks load the same data no matter where the kernel was started.

A rollout dump is a dict of arrays shaped (timestep, env); a "run" is one
policy's dump for one sensor set on one terrain.
"""

from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# Terrain key -> filename suffix in HEBB-<sensors>-<policy>-bend-<suffix>.npy.
# The keys are what the figures index by; display names live in paper_style.
TERRAIN_FILE = {
    "flat":     "flat",
    "rough":    "rough",
    "morph":    "morph",
    "slope10":  "slope10",
    "slope_10": "slope-10",
    "mass1000": "mass",
}

N_POLICIES = 5


def _load(subdir, name):
    return np.load(DATA_ROOT / subdir / f"{name}.npy", allow_pickle=True).item()


def load_terrain(sensors, terrain, *, n_policies=N_POLICIES, subdir="data_2026"):
    """-> {sensor: [run for each policy]} for one terrain."""
    suffix = TERRAIN_FILE[terrain]
    return {s: [_load(subdir, f"HEBB-{s}-{p}-bend-{suffix}") for p in range(n_policies)]
            for s in sensors}


def load_terrains(sensors, terrains=None, *, n_policies=N_POLICIES, subdir="data_2026"):
    """-> {terrain: {sensor: [run per policy]}} for every requested terrain."""
    terrains = list(TERRAIN_FILE) if terrains is None else terrains
    return {t: load_terrain(sensors, t, n_policies=n_policies, subdir=subdir)
            for t in terrains}


def load_ablations(ablations, base_sensors, *, n_policies=N_POLICIES, subdir="dropout"):
    """Sensor-dropout runs plus the un-ablated baseline.

    -> ({ablation: [run per policy]}, [baseline run per policy])
    """
    dropped = {a: [_load(subdir, f"HEBB-{base_sensors}-{p}-bend-drop_{a}")
                   for p in range(n_policies)]
               for a in ablations}
    baseline = [_load(subdir, f"HEBB-{base_sensors}-{p}-bend-flat") for p in range(n_policies)]
    return dropped, baseline


# ── Reduction: rollout -> one number per environment ────────────────────────
def max_distance(run):
    """Furthest forward displacement each environment reached. -> (n_env,)"""
    return np.ravel(np.max(run["pos_x"], axis=0)).astype(float)


def pooled(runs_per_policy):
    """Concatenate a metric across policies. -> (n_policies * n_env,)"""
    return np.concatenate([max_distance(r) for r in runs_per_policy])


def by_sensor(terrain_runs, *, policy=None):
    """Max distance per sensor for one terrain.

    `policy=None` pools all policies (n=150); an int selects one (n=30).
    """
    if policy is None:
        return {s: pooled(runs) for s, runs in terrain_runs.items()}
    return {s: max_distance(runs[policy]) for s, runs in terrain_runs.items()}


def normalize(per_sensor, *, max_score=10.0):
    """Scale so the best single rollout in this terrain scores `max_score`.

    Normalising within a terrain is what makes scores comparable across
    terrains whose absolute distances differ by a factor of two.
    """
    top = max(v.max() for v in per_sensor.values())
    return {s: v * max_score / top for s, v in per_sensor.items()}
