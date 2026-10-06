import numpy as np

# ----- your indices -----
pos_idx    = np.arange(0, 19)
vel_idx    = np.arange(19, 19*2)
action_idx = np.arange(19*2, 19*3)
IMU_idx    = np.arange(57, 60)
fc_idx     = np.arange(60, 64)

MODALITY_BITS = [
    ("pos",    1,  pos_idx),
    ("vel",    2,  vel_idx),
    ("action", 4,  action_idx),
    ("imu",    8,  IMU_idx),
    ("fc",     16, fc_idx),
]

def decode_sensor_mask(mask: int, keep_order: bool = True):
    """
    mask: integer bitmask from bash
    keep_order=True:  MODALITY_BITS
    overlap: 
    return:
      selected_names: list[str]
      selected_indices: np.ndarray[int]
    """
    selected_names = []
    seen = set()
    out = []

    for name, bit, idx in MODALITY_BITS:
        if mask & bit:
            selected_names.append(name)
            if keep_order:
                for i in idx.tolist():
                    if i not in seen:
                        seen.add(i)
                        out.append(i)
            else:
                out.append(idx)

    if keep_order:
        selected_indices = np.array(out, dtype=int)
    else:
        # union but may reorder (sorted unique)
        selected_indices = np.unique(np.concatenate(out)) if out else np.array([], dtype=int)

    return selected_names, selected_indices
