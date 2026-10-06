# Methods — teacher-student distillation and sensor-importance analysis

Experimental conditions for `STUDENT-dagger-0`: 6 independent students (flat,
rough, slope_up, slope_down, morph, mass), each distilled from the same
Hebbian teacher via pure DAgger, then evaluated and analysed with integrated
gradients. Numbers below are pulled from the actual run logs and checkpoint
metadata, not just script defaults.

## 1. Teacher

| | value |
|---|---|
| algorithm | Evolution Strategy (OpenES), `HEBB-pos_vel_action_imu_fc-0` |
| architecture | Hebbian plastic network, **64 → 128 → 64 → 19**, tanh |
| plasticity params | per-layer A, B, C, D, learning-rate tensors (Hebbian rule), `init_noise=0.04`, `norm_mode='var'`, `std_init_param=0.025` |
| population size | 1024 |
| ES hyperparameters | `learning_rate=0.1` (decay 0.9999), `sigma_init=0.1` |
| training epochs | 100 used (checkpoint `model_99.pickle`) |
| checkpoint | `logs/es/bend/hebb/HEBB-pos_vel_action_imu_fc-0/model/model_99.pickle` |
| behaviour | deterministic given fixed evolved parameters, but *stateful*: plastic weights update every forward pass within an episode; reset at episode boundaries during distillation |

## 2. Student

| | value |
|---|---|
| architecture | plain MLP, **64 → 128 → 64 → 19** (identical shape to teacher, no plasticity) |
| activation | tanh (hidden and output — output tanh keeps commands in [-1, 1]) |
| initialisation | orthogonal weight init, gain = √2, zero bias |
| parameters | ~19.2K (64×128+128 + 128×64+64 + 64×19+19) |
| implementation | `scripts/ES/utils/student_net.py::MLPStudent` |

## 3. Distillation method — pure DAgger

No reward, no critic, no policy gradient. One loss:

```
loss = MSE( student(state), teacher(state) )
```

| parameter | value |
|---|---|
| iterations | **400** per terrain (all 6 runs completed iteration 399/400) |
| parallel environments | **1024** |
| steps per iteration | **64** (env-steps collected before each supervised update) |
| total env-steps per terrain | 400 × 64 × 1024 ≈ **26.2M** |
| aggregated dataset (buffer) | capacity **2,000,000** states, ring buffer, fills after ~31 iterations |
| supervised updates per iteration | **40** gradient steps |
| batch size | **4096** |
| optimizer | AdamW, lr = **1e-3** (fixed, no decay), weight_decay = 0 |
| loss function | MSE |
| gradient clipping | max norm **1.0** |
| **teacher_iters** (pure-teacher warm-up) | **60** iterations, β = 1.0 (teacher drives all 1024 envs, pure behaviour cloning) |
| β schedule after warm-up | `β = max(0, 1.0 × 0.97^(iter − 60))` — half-life ≈ 23 iterations |
| β floor | **0.0** (fully student-driven by ~iteration 260) |
| environment assignment | per-environment for the whole rollout (not per-step) — each trajectory has one consistent behaviour policy |
| joint order | remapped to `cfg.actuated_joint_names` (`--joint_order cfg`) to match the order the teacher was trained under |

**Seeds used per terrain** (`--seed -1`, i.e. randomly drawn, one per run):

| terrain | seed | slope_deg |
|---|---|---|
| flat | 4976 | — |
| rough | 4661 | — |
| slope_up | 334 | −10.0° |
| slope_down | 7890 | +10.0° |
| morph | 758 | — |
| mass | 5921 | — |

**Independent students** — no weight sharing, no chaining, no rehearsal buffer
across terrains. Each is a separate `train_student.py` process, its own
checkpoint under `logs/distill/STUDENT-dagger-0-<terrain>/model/student_latest.pt`.

## 4. Physics / control timing

| | value |
|---|---|
| simulation dt | 1/200 s |
| control decimation | 4 → control step = 0.02 s |
| episode length | 30 s = **1500 control steps** |
| iteration-to-episode ratio | 64 env-steps/iteration ≈ 1.28 s of sim time; ~23 iterations per full episode |

## 5. Evaluation (data collection for IG)

Actor-only, deterministic (no exploration noise, no teacher in the loop) —
`runner/distill/eval_student.sh` → `scripts/ES/play_student.py`.

| parameter | value |
|---|---|
| environments | **30** |
| episodes | **1** |
| steps per episode | **1000** (`EPISODE_LENGTH_TEST` from `es_cfg.yaml`) |
| total transitions collected per terrain | 1000 × 30 = **30,000** (state, action, …) tuples |
| seed | 42 (fixed, identical across all terrains for comparability) |
| policy | student's deterministic mean action (`.act()`), no sampling |
| saved fields | `state` (64-d obs), `action` (19-d), `reward`, `dof_pos`/`dof_vel`, `torque`, `vel_w`/`vel_b`, `fc`, `pos_x`/`pos_y`, `yaw`, `joint_names` |

## 6. Integrated Gradients — feature-importance settings

| parameter | value |
|---|---|
| states analysed per terrain | **20,000**, randomly subsampled from the 30,000 collected transitions (`--samples 20000`) |
| baseline | **zeros** (scaled-midpoint observation) |
| integration steps | **64** (Riemann midpoint rule along the straight path baseline → input) |
| output reduction | **abs_sum** — sum \|attribution\| across all 19 joint outputs, so a sensor counts if it moves *any* joint |
| completeness error (sanity check) | ranged **3.7e-4 to 9.2e-4** across terrains (flat lowest, mass highest) — acceptable but not tight; consider 128–256 steps for a final write-up |
| grouping | 64 sensors → 5 modalities: pos (19) / vel (19) / action (19) / imu (3) / fc (4) |

## Caveats

- **No per-timestep saliency.** Subsampling pools states across all 30 envs ×
  1000 steps and shuffles them before running IG, so the reported importance
  is an average over the whole rollout — it cannot show how importance shifts
  within an episode (e.g. spiking right before a stumble).
- **Importance is behaviour-conditioned, not task-conditioned.** The 20,000
  states come from *this* student's own deterministic rollout. A student
  trained with a different seed, β schedule, or algorithm (e.g. PPO instead of
  DAgger) walks differently, visits different states, and could report a
  different sensor ranking on the identical terrain — the numbers describe
  "what matters given how this policy behaves," not an algorithm-independent
  property of the terrain.
