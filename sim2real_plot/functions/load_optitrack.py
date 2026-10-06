import numpy as np
import glob
import pandas as pd
import os
# ---------------------------------------------------------------- OptiTrack
# Exported takes have 6 preamble lines (metadata, blank, Type, Name, ID,
# Rotation/Position) before the real header row:
#   Frame,Time (Seconds),X,Y,Z,X,Y,Z   -> rotation XYZ then position XYZ
# Length units are millimetres, Y is up, coordinate space is Global.
OPTITRACK_COLUMNS = ['frame', 'time',
                     'rot_x', 'rot_y', 'rot_z',
                     'pos_x', 'pos_y', 'pos_z']


def load_optitrack_csv(CASE, FOLDER, MODEL, trial, BASE_PATH, to_meters=True):
    path = os.path.join(BASE_PATH, FOLDER, 'trajectory', CASE, f'{CASE}-{MODEL}-{trial}.csv')

    df = pd.read_csv(path, skiprows=6)
    if df.shape[1] != len(OPTITRACK_COLUMNS):
        raise ValueError(f'{path}: expected {len(OPTITRACK_COLUMNS)} columns, got {df.shape[1]}')
    df.columns = OPTITRACK_COLUMNS
    df = df.astype(float)
    df['frame'] = df['frame'].astype(int)
    if to_meters:
        df[['pos_x', 'pos_y', 'pos_z']] /= 1000.0
    return df


def load_multi_optitrack(CASE, FOLDER, MODEL, TRIALS, TRAJ_BASE):
    dataset = np.array([])
    for i in range(len(TRIALS)):
        df = load_optitrack_csv(CASE, FOLDER, MODEL, TRIALS[i], TRAJ_BASE)
        separate_cols = {col: df[col].to_dict() for col in df.columns}
        dataset = np.append(dataset, separate_cols)
        
    return dataset

def _as_frame(take):
    """Takes come out of load_multi_optitrack as {col: {idx: val}}; accept a DataFrame too."""
    return take.copy() if isinstance(take, pd.DataFrame) else pd.DataFrame(take)


def set_initial_position(dataset, n_ref=1, zero_height=False):
    out = []
    for take in dataset:
        df = _as_frame(take)

        valid = df[['pos_x', 'pos_y', 'pos_z', 'rot_y']].notna().all(axis=1)
        if not valid.any():
            raise ValueError('take has no frame with both position and rotation')
        ref = df.loc[valid].iloc[:n_ref]

        x0, y0, z0 = ref['pos_x'].mean(), ref['pos_y'].mean(), ref['pos_z'].mean()
        psi0 = np.deg2rad(ref['rot_y'].mean())

        # Still in the capture frame here, just translated to the start and
        # turned about the vertical so the initial heading is 0.
        dx, dz = df['pos_x'] - x0, df['pos_z'] - z0
        c, s = np.cos(psi0), np.sin(psi0)
        ox, oz = dx * c - dz * s, dx * s + dz * c
        oy = df['pos_y'] - y0 if zero_height else df['pos_y']

        # Capture-frame axes, kept exactly as OptiTrack reports them: no axis
        # is negated, reordered, or remapped. x_rel / z_rel span the ground
        # plane, y_rel is height above the floor (Y is up).
        df['x_rel'], df['y_rel'], df['z_rel'] = ox, oy, oz

        # np.unwrap propagates NaN, so unwrap across the tracked samples only
        # and leave dropped frames as NaN rather than poisoning everything after.
        yaw_deg = df['rot_y'].to_numpy(dtype=float)
        tracked = np.isfinite(yaw_deg)
        yaw = np.full(yaw_deg.shape, np.nan)
        yaw[tracked] = np.unwrap(np.deg2rad(yaw_deg[tracked]))
        # Net turn since the start, radians, positive to the left.
        df['yaw_rel'] = yaw - psi0

        out.append(df)

    # a plain list, not np.array: stacking DataFrames of equal length would
    # collapse them into one numeric array
    return out
