# Cooperative T-Shape Transport Using Multi-Agent Reinforcement Learning

**A MuJoCo-based multi-agent reinforcement learning framework for cooperative object transport using Multi-Agent Proximal Policy Optimization (MAPPO).**

This project explores how multiple autonomous agents can learn to coordinate their actions to transport a shared rigid object toward a target location.

Inspired by cooperative geometric puzzle-solving experiments involving ants and humans, the project adapts collective object transportation into a physics-based multi-agent reinforcement learning environment.

The primary objective is to investigate **emergent cooperative behavior**, where multiple agents learn to coordinate through physical interactions without explicitly programmed individual roles.

The current implementation supports **Easy, Medium and Hard arenas with configurable agent counts and global actor information**. Medium adds one divider; Hard adds two dividers with aligned openings. Agents start at random exterior positions near the object. The obstacle layouts and their target positions are fixed. All three policies use the same observation features and reward. The framework includes task diagnostics, staged curricula, actor transfer, held-out checkpoint benchmarks, and independent agent-count/reward experiments. Limited-range observations and held-out-layout benchmarks remain research extensions. Successful transport alone does not establish emergent coordination.

---

## 1. Project Overview

The default simulation consists of four circular robotic agents interacting with a passive rigid object inside a bounded arena.

Each agent independently applies forces along the X and Y axes. The transported object cannot be directly controlled and must be moved through physical contact between the agents and the object.

The environment supports simultaneous multi-agent actions, cooperative rewards, continuous control, and reinforcement learning through MAPPO.

### Key Features

- Physics-based simulation using MuJoCo
- Four independently controlled robotic agents
- Passive rigid-body object transportation
- Continuous force-based control
- Contact and collision simulation
- PettingZoo Parallel API integration
- Cooperative shared reward
- Shared actor and centralized critic architecture
- Multi-Agent Proximal Policy Optimization (MAPPO)
- Generalized Advantage Estimation (GAE)
- PPO clipped objective
- Gaussian policy with `tanh` action transformation
- Training, evaluation, checkpointing, and resume functionality
- Step-based training budgets, evaluation, checkpointing, and 1080p videos
- TensorBoard graphs for rewards, success, episode length, losses, and throughput
- Two retained training checkpoints: `best.pt` and `latest.pt`
- Physics diagnostics and environment validation

## 2. System Architecture

```text
                  +------------------------+
                  |     MAPPO / PyTorch    |
                  |                        |
                  |  Shared Actor          |
                  |  Centralized Critic    |
                  +-----------+------------+
                              |
                         Agent Actions
                              |
                              v
                  +------------------------+
                  | PettingZoo ParallelEnv |
                  |                        |
                  | Observations           |
                  | Cooperative Rewards    |
                  | Episode Management     |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  |     Physics Adapter    |
                  |                        |
                  | Action Scaling         |
                  | Actuator Mapping       |
                  | State Extraction       |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  |         MuJoCo         |
                  |                        |
                  | Rigid-Body Dynamics    |
                  | Contact Resolution     |
                  | Collision Detection    |
                  +-----------+------------+
                              |
                         Updated State
                              |
                              v
                     Environment Step
```

### Technology Stack

| Technology | Purpose                                        |
| ---------- | ---------------------------------------------- |
| Python     | Main programming language                      |
| MuJoCo     | Rigid-body physics simulation                  |
| PettingZoo | Multi-agent environment interface              |
| Gymnasium  | Action and observation space definitions       |
| PyTorch    | Neural network implementation and optimization |
| MAPPO      | Cooperative multi-agent reinforcement learning |

## 3. Simulation Environment

### Robotic Agents

The default environment contains four circular robotic agents; `num_agents` in the environment YAML supports 1–16, with experiment matrices for 2, 4, 8 and 16.

Each agent is independently controlled through two force actuators corresponding to the X and Y directions.

| Parameter              | Value         |
| ---------------------- | ------------- |
| Number of agents       | 4 default; 2/4/8/16 experiments |
| Agent radius           | 0.03 m        |
| Agent diameter         | 0.06 m        |
| Agent mass             | 0.25 kg       |
| Control dimensions     | 2             |
| Maximum actuator force | ±3 N per axis |
| Action range           | [-1, 1]       |

The agents interact with the transported object exclusively through physical contact.

### Transported Object

The transported object is an asymmetric rigid body consisting of a central elongated structure and two horizontal crossbars of different lengths.

Its movement is constrained to three planar degrees of freedom:

- Translation along the X axis
- Translation along the Y axis
- Rotation around the vertical axis (yaw)

The object has a total mass of **2 kg** and an overall footprint of **0.6 x 1.2 m**. The large crossbar is 0.6 m wide; the smaller crossbar is 0.3 m wide. Three box geometries form one rigid body. These dimensions approximate the reference image; they are not measurements from the original experiment.

Joint damping and friction loss are used to model movement resistance and reduce excessive sliding.

Easy and Medium measure **5.8 x 5.8 m** internally. Easy has boundary walls only. Medium adds a fixed vertical divider at `x=0`, 0.2 m thick, with a **0.8 m opening** spanning `y=-0.4` to `0.4`. The object's 1.2 m span blocks some orientations; rotating it by 90 degrees gives a 0.6 m span across the opening.

Hard measures **9.0 x 5.8 m**, with dividers at `x=-1.1` and `x=1.1`. Both 0.8 m openings are centered at `y=0`, spanning `[-0.4, 0.4]`. The middle room is 2.0 m wide. After suitable rotation and alignment, the object can cross both dividers along the same line. These courses are inspired by the reference, not measured replicas of the original maze.

Medium's target center stays at **`(1.8, 1.5)` m** and Hard's at **`(3.25, 1.5)` m** on every episode. Objects start randomly in the leftmost room, so successful transport must cross all dividers. Agents start at independently randomized exterior positions around the object, with a 0.04–0.15 m surface-to-surface gap and no assigned pushing roles.

Bodies remain at fixed height with zero gravity; the floor is visual only. Support resistance is approximated by joint damping and dry friction rather than physical floor contact.

### Physics Configuration

| Parameter                     | Value              |
| ----------------------------- | ------------------ |
| Physics engine                | MuJoCo             |
| Physics timestep              | 0.001 s            |
| Physics substeps per action   | 50                 |
| Control interval              | 0.05 s             |
| Control frequency             | 20 Hz              |
| Maximum episode duration      | Easy: 25 s; Medium: 75 s; Hard: 125 s |
| Maximum environment steps     | Easy: 500; Medium: 1,500; Hard: 2,500 |
| Contact response              | `solref="0.002 1"` |
| Contact penetration threshold | 5 mm               |

The simulation uses a smaller physics timestep to improve contact resolution while maintaining a consistent control frequency.

The object uses XY damping of 4 N·s/m and yaw damping of 0.8 N·m·s/rad, with dry friction of 1.5 N per translation axis and 0.16 N·m on yaw. Agent damping is 1 N·s/m per axis. Easy uses [arena.xml](assets/mujoco/arena.xml) and [environment.yaml](configs/environment.yaml). Medium uses [arena_medium.xml](assets/mujoco/arena_medium.xml) and [environment_medium.yaml](configs/environment_medium.yaml), with the same object/agent geometry, mass, resistance, actuator limits, and contact settings.

Hard uses [arena_hard.xml](assets/mujoco/arena_hard.xml) and [environment_hard.yaml](configs/environment_hard.yaml), with the same object/agent physics and unchanged reward weights. The 2,500-step horizon is an initial budget for the longer course, not a demonstrated learned completion time.

## 4. Multi-Agent Environment

The environment is implemented using the **PettingZoo Parallel API**, allowing all agents to execute actions simultaneously.

### Action Space

Each agent produces a continuous two-dimensional action:

$$
\mathbf{a}_i = [a_{x,i}, a_{y,i}], \qquad a_{x,i}, a_{y,i} \in [-1,1].
$$

Actions are converted into actuator forces:

$$
F_{x,i} = F_{\max} a_{x,i}, \qquad F_{y,i} = F_{\max} a_{y,i}.
$$

With the current symmetric XML actuator limits, $F_{\max}=3\,\mathrm{N}$ per axis. The maximum force vector magnitude is therefore $3\sqrt{2}\,\mathrm{N}$, approximately 4.24 N. Zero action means zero motor force. Finite actions outside the allowed range are clipped; invalid shapes or non-finite actions are rejected.

### Observation Space

With four agents, each actor receives **34 float32 features**. Each agent has its own observation vector, with relative positions measured from that agent and vectors expressed along world XY axes:

| Actor features | Dimensions |
| --- | ---: |
| Own position and velocity | 4 |
| Object position relative to the agent | 2 |
| Sine/cosine of object yaw | 2 |
| Object linear velocity and yaw rate | 3 |
| Goal position relative to the agent and goal half-size | 4 |
| Other agents' relative positions and velocities | 12 |
| Four wall clearances | 4 |
| Contact with the object | 1 |
| Remaining-time fraction and goal-hold fraction | 2 |

With four agents, the centralized critic receives a separate **33-feature global state** containing object motion, goal geometry, all agent positions/velocities, arena bounds, and episode fractions. Feature order is defined in [observations.py](src/envs/observations.py).

Actor execution does not call the critic. However, actor information is currently global and has no sensor-range masking. Limited local perception and communication experiments are not implemented yet.

All difficulty levels use the same feature definitions. With N agents, actor size is `18 + 4N` and critic state size is `17 + 4N`; the four-agent schema remains 34/33. Only the lengths of agent/neighbor arrays change; no obstacle features are added. Medium and Hard have no added obstacle features or internal-wall range sensors. The four wall-clearance features still refer only to the outer arena boundaries. Their fixed dividers must be learned through interaction and the existing world-position information.

### Reward Function

The environment uses a shared cooperative reward.

The reward encourages agents to move the transported object toward the target while accounting for episode duration and physical failure conditions.

Let $d_t$ be the Euclidean distance between the object's center of mass and the goal center. Let $g_t$ indicate success and $f_t$ indicate physical failure; each indicator is 0 or 1. Progress is suppressed on a failing transition:

$$
\Delta d_t =
\begin{cases}
d_{t-1} - d_t, & f_t = 0, \\
0, & f_t = 1.
\end{cases}
$$

The default reward is:

$$
r_t = 2\Delta d_t - 0.001 + 20g_t - 20f_t.
$$

The same calculation in plain text, for Markdown viewers without math rendering:

```text
progress = 0 if physics_failure else previous_distance - current_distance
reward = 2 * progress - 0.001 + 20 * success - 20 * physics_failure
```

| Component | Default setting |
| --- | ---: |
| Progress weight | 2.0 |
| Time penalty per environment decision | 0.001 |
| Success bonus | 20.0 |
| Physical failure penalty | 20.0 |

All agents receive the same team reward. Training stores it once per team transition; it is not multiplied by the agent count. Episode return is the sum of these rewards over an episode. A stationary episode has a return near -0.5 for Easy, -1.5 for Medium, or -2.5 for Hard at their full horizons; moving 1 cm closer in one ordinary step gives 0.019.

There is currently no reward for an agent merely approaching/touching the object, no energy penalty, and no ordinary collision penalty. Valid pushing contact is necessary. Wall contacts and control effort are logged as diagnostics; a contact triggers the failure penalty only if it leads to a physical safety violation. The existing `wall_collision_events` and `wall_contact_physics_steps` metrics include all internal dividers. See [rewards.py](src/envs/rewards.py) for the implementation.

Medium and Hard use the same Euclidean progress and reward weights as Easy. There are no rotation bonuses, waypoints, path-planner rewards, or extra maneuver incentives. A temporarily increasing goal distance can reduce reward even when the maneuver is useful; solving these baselines is an empirical training question.

### Episode Termination

An episode ends when:

1. **Success (`terminated=True`):** all corners of all three object geometries fit inside the goal rectangle, linear speed is at most 0.05 m/s, and absolute yaw rate is at most 0.1 rad/s for five consecutive decisions (0.25 s by default).
2. **Time limit (`truncated=True`):** the configured limit is reached without success or physical failure: 500 decisions for Easy, 1,500 for Medium, 2,500 for Hard.
3. **Physical failure (`terminated=True`):** excessive penetration, leaving the arena, non-finite state, MuJoCo warnings, excessive speed, or an unexpected automatic simulator reset is detected.

The goal half-size is `(0.75, 0.65)` m. Its center is randomized for Easy, fixed at `(1.8, 1.5)` m for Medium and `(3.25, 1.5)` m for Hard. There is no single required yaw: any orientation that fits and satisfies the settling criteria is allowed. Diagnostics run after every physics substep. The linear speed threshold of 5 m/s and object yaw-rate threshold of 10 rad/s are failure checks, not velocity controllers.

### Random Resets with Agents Near the Object

Every newly started training process, including resume, randomizes object pose and agent positions. All agents start **0.04–0.15 m from the object's actual surface**. Easy also randomizes its target; Medium and Hard keep their configured target fixed. No new CLI flag is needed.

| Reset variable | Current distribution |
| --- | --- |
| Object yaw | Full `[-pi, pi]` range |
| Object XY | Across the feasible arena for Easy; inside the starting end room for Medium/Hard |
| Each agent | Random exterior face and position on the actual compound object, with a surface gap sampled from `[0.04, 0.15]` m |
| Easy goal center | Random across the arena, with the complete rectangle at least 0.15 m from walls |
| Medium / Hard goal center | Fixed at `(1.8, 1.5)` / `(3.25, 1.5)` m |
| Object and agent outer-wall margin | At least 0.15 m |
| Agent/agent and internal-wall clearance | At least 0.04 m |
| Minimum object-to-goal center distance | 1.5 m; already-solved starts are rejected |

The sampler uses exterior faces of the three object boxes. Candidates on hidden internal faces, too close to another agent, or beside a wall are rejected. Each agent can appear on different sides across episodes; the randomized sampler does not assign agents fixed pushing slots. Distances refer to the object footprint, not its body center or enclosing rectangle. Object position and orientation remain broadly randomized; the reduced randomization concerns agent placement.

Tune the gap in `agent_spawn_gap: [0.04, 0.15]` in the environment YAML. Its minimum must be at least `spawn_clearance`. Seeded resets reproduce the same object, goal, nearby agents and action-space sampling. Reset/terminal info records the actual initial state and `reset_distribution: near_object`.

The old arena-wide agent sampler, legacy small-noise reset branch, and `object_position_noise`, `object_yaw_noise`, `agent_position_noise` fields were removed. Fixed deterministic starts remain available for physics/scripted diagnostics with `--no-randomize`; they are not a training mode.

Old checkpoint configurations are translated to the current near-object reset instead of keeping old sampling implementations in the environment. Evaluation records `reset_distribution_migrated` and `saved_reset_distribution`, so new scores are not mistaken for the old task distribution. Medium/Hard goals are fixed after migration too. Resume retains weights, optimizer state and cumulative steps, but clears both input normalizers and the rolling episode-statistics window when the saved distribution changes. The manifest records the migration.

Checkpoints still contain their original MuJoCo XML. Resuming an older Hard checkpoint cannot silently replace its course geometry. Training rejects saved Hard XML whose openings are not aligned at `y=0`. Start a **new Hard run** to train the corrected aligned course. To evaluate old weights specifically on the current course and nearby-agent resets, use the existing override:

```bash
python scripts/evaluate.py results/mappo_hard/checkpoints/best.pt --env-config configs/environment_hard.yaml --device auto --episodes 50 --seed 200000 --json results/mappo_hard/eval_aligned_near_object.json
```

An already-running process keeps its previous environment until restarted/resumed with the updated code.

## 5. MAPPO Algorithm

This project implements **Multi-Agent Proximal Policy Optimization (MAPPO)** for cooperative continuous control.

This implementation combines PPO's clipped objective with a shared actor and centralized team-value estimation.

### Shared Actor

All agents share the same actor network parameters.

The actor receives observations and produces a probability distribution over continuous actions.

A Gaussian policy is used to sample actions, followed by a `tanh` transformation to keep actions within the allowed range.

For four agents the network is `34 -> 128 -> 128 -> 2`, with Tanh hidden activations and a learned, state-independent log standard deviation. The actor stores pre-tanh samples and includes the transformation's log-Jacobian in action likelihoods.

### Centralized Critic

The centralized critic estimates the expected cumulative reward using global state information.

This allows the critic to evaluate the collective state of the environment during training.

For four agents the critic is `33 -> 256 -> 256 -> 1`, with Tanh hidden activations and one scalar value per team state.

### Generalized Advantage Estimation

GAE estimates the advantage of selected actions. Let $T_t$ indicate true termination and $U_t$ indicate time-limit truncation. The implementation uses separate masks for bootstrapping and carrying advantages:

$$
\delta_t = r_t + \gamma(1-T_t)V_{\phi}(s_{t+1}) - V_{\phi}(s_t),
$$

$$
\hat{A}_t = \delta_t + \gamma\lambda(1-T_t)(1-U_t)\hat{A}_{t+1}.
$$

True termination does not bootstrap. A time limit bootstraps from the final state **before reset**, while advantage accumulation stops at either kind of episode boundary. A rollout cut also uses the next-state value. The same team advantage is used for the N actor samples at that time step. See [gae.py](src/algorithms/gae.py).

### PPO Objective

The actor is optimized using the clipped PPO objective:

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb{E}_{t,i}\left[
\min\left(
\rho_{t,i}(\theta)\hat{A}_t,
\operatorname{clip}(\rho_{t,i}(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right]
$$

where:

$$
\rho_{t,i}(\theta)=
\frac{\pi_{\theta}(a_{t,i}\mid o_{t,i})}
{\pi_{\theta_{\mathrm{old}}}(a_{t,i}\mid o_{t,i})}.
$$

The code minimizes the negative clipped objective with an entropy bonus. The critic uses a value prediction loss with optional value clipping. Input normalization statistics remain frozen during rollout collection and PPO epochs, then update for the next rollout. Evaluation uses the stored statistics without changing them.

Equations use `$$ ... $$` display blocks and `$ ... $` inline expressions. View them in a math-enabled Markdown preview; see [GitHub's math formatting guide](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).

In VS Code, open the Markdown preview with `Ctrl+Shift+V` and ensure the built-in Markdown Math extension is enabled if formulas appear as raw text.

### Training Features

- Shared actor network
- Centralized critic network
- Gaussian action distribution
- `tanh` action squashing
- PPO clipping
- Generalized Advantage Estimation
- Advantage normalization
- Gradient clipping
- Rollout buffer
- Checkpoint saving
- Training resumption
- Policy evaluation

### Default Hyperparameters

Defaults come from [mappo.yaml](configs/mappo.yaml), not from guaranteed optimal settings:

| Setting | Default |
| --- | ---: |
| Actor / critic learning rate | 0.0003 / 0.0003 |
| Discount / GAE lambda | 0.99 / 0.95 |
| PPO clip / value clip | 0.2 / 0.2 |
| PPO epochs / minibatches | 5 / 1 |
| Rollout length | 1,024 team decisions |
| Entropy coefficient / value coefficient | 0.01 / 0.5 |
| Maximum gradient norm | 0.5 |
| Target KL for early stopping | 0.03 |
| Evaluation interval / episodes | Every 50,000 team steps / 5 episodes |
| Latest-checkpoint interval | Every 10,000 team steps, at video milestones, and on normal completion |
| Video interval | Every 100,000 team steps |
| Rolling episode-statistics window | 100 completed episodes |
| Raw reward sampling interval | Every 100 team steps; set to 1 for every step |

A rollout of 1,024 team decisions provides 4,096 actor samples and 1,024 team-value samples. Training currently uses one environment process.

Step schedules are checked **after PPO updates**, at rollout boundaries. The 50,000-step evaluation threshold is reached at 50,176 with the default rollout; the 10,000-step checkpoint threshold is reached at 10,240; the 100,000-step video threshold is reached at 100,352. One evaluation/video is produced for the thresholds crossed by an update, rather than repeating identical evaluations if several thresholds were crossed. Final evaluation and checkpoint saving also run when the exact step budget is exhausted.

## Research Workflows

### Task Diagnostics

Episode info/logs include `diagnostics`: signed footprint margin, current/minimum footprint error, inside-but-unsettled steps, center crossings versus complete object clearance at each divider, first complete-clear step, and per-agent contact duration. Contact durations integrate contact flags at every physics substep. They measure contact, not useful mechanical work or emergent coordination.

Terminal failure categories are `gate_not_reached`, `gate_blocked`, `contact_lost`, `goal_footprint_outside`, `goal_not_settled`, or `physics_failure`; success is reported separately. Gate blockage takes precedence when a divider remains uncleared, while `contact_lost_at_end` retains the secondary contact information. Geometric fields on a physics failure retain the last stable geometry; `failure_details` retains the actual failing state.

These measurements do not enter actor/critic inputs or change the baseline reward. TensorBoard adds `rollout/footprint_error_m_mean`, `rollout/footprint_margin_m_mean`, `rollout/gate_clear_fraction_mean`, failure-category rates, and `contact/agent_<i>_time_s` at episode completion.

### Curriculum Training

```bash
python scripts/train.py --env-config configs/environment.yaml --curriculum configs/curricula/easy.yaml --device cuda --total-steps 2000000 --run-dir results/easy_curriculum
```

Each curriculum has three stages:

| Arena | Stage 1 | Stage 2 | Final stage |
| --- | --- | --- | --- |
| Easy | Goal distance 0.8–1.2 m, yaw ±0.15 rad, 250-step horizon | Distance 1.2–2.0 m, yaw ±0.75 rad, 400-step horizon | Original full task |
| Medium | Initial yaw 1.42–1.72 rad, fixed target `(1.8, 0)`, horizon 800 | Full yaw, same centerline target, horizon 1,200 | Original target `(1.8, 1.5)` and horizon 1,500 |
| Hard | Initial yaw 1.42–1.72 rad, fixed target `(3.25, 0)`, horizon 1,500 | Full yaw, same centerline target, horizon 2,000 | Original target `(3.25, 1.5)` and horizon 2,500 |

Agent positions remain random near the object at every stage. Success still requires full containment and settling. Definitions live in `configs/curricula/`; the same bounded pose sampler handles every stage. No scripted action or waypoint controller is used by the learner.

Every 50,000 team steps, a separate stage-validation pool of 20 seeds starting at 150000 is evaluated. Promotion requires at least 70% success on two consecutive checks and 50,000 steps in that stage. These are configurable engineering thresholds, not evidence that a policy has generalized. The final stage restores the complete baseline task; satisfying its threshold marks the curriculum complete and stops extra stage checks.

Full-task evaluation (`eval_seed=100000`, five seeds by default), `best.pt` selection, and videos always use the full final task, even during an easier stage. Stage reports are appended to `evaluations.jsonl` as `kind: curriculum`; they never overwrite the full-task latest/best evaluation snapshots. Curriculum checks add simulation cost in addition to normal monitoring.

Stage switches occur after PPO updates. A partial episode is discarded and its seed/step count recorded in `curriculum_transitions.jsonl`; it is not counted as a completed episode or a failure. The rolling episode window clears at a switch. Curriculum definitions, active stage, streak, start step and completion state are checkpointed. Resume uses the saved definition even if the original YAML path is unavailable. No stage is forced merely because the step budget ends.

### Actor Transfer Between Arenas

After verifying an Easy policy on the full task:

```bash
python scripts/train.py --env-config configs/environment_medium.yaml --curriculum configs/curricula/medium.yaml --init-actor results/easy_curriculum/checkpoints/best.pt --device cuda --total-steps 2000000 --run-dir results/medium_curriculum_transfer
```

`--init-actor` starts a new experiment and copies actor weights **and actor input-normalization statistics**, preserving its initial input mapping. The critic, critic normalization, both optimizers, counters and RNG are new. Transfer requires the same agent count, feature schema, actor architecture and normalization settings. Different agent counts are trained independently. `--init-actor` cannot be combined with `--resume`; resume restores the original task, stage and optimizers.

### Held-Out Best/Latest Benchmark

```bash
python scripts/benchmark.py results/easy_curriculum/checkpoints/best.pt results/easy_curriculum/checkpoints/latest.pt --episodes 50 --device auto --output results/easy_curriculum_benchmark
```

Both deterministic and stochastic modes run on the same seeds starting at 200000. Updated training excludes `[200000, 300000)` from its episode-seed sequence, along with full-task and curriculum validation pools. The benchmark rejects overlap with saved validation pools. An old checkpoint may have trained on other seed schedules; verify its provenance before calling its results held-out.

Output is bounded to a manifest, `episodes.jsonl`, `summary.csv`, and `summary_by_group.csv`. Per-policy success intervals use the 95% Wilson binomial formula. Group summaries report mean and sample standard deviation across policies, with the number of unique training seeds explicitly shown; one policy has no between-seed standard deviation. Grouping separates task/configuration, training protocol, reward, agent count, mode and best/latest. A confidence interval on a checkpoint selected using five repeated validation seeds is not a generalization estimate.

The benchmark uses full saved task settings; `--env-config` explicitly evaluates an alternative task. It does not record videos by default. Use `scripts/evaluate.py` for selected recordings.

### Agent-Count Experiments

```bash
python scripts/run_experiments.py configs/experiments/agent_count.yaml --output results/agent_count --dry-run
python scripts/run_experiments.py configs/experiments/agent_count.yaml --output results/agent_count --device cuda
```

The matrix declares 2, 4, 8 and 16 agents, five training seeds (11, 22, 33, 44, 55), and 2 million team steps per policy: **20 independent runs / 40 million team steps**. The launcher runs them sequentially, records status and best validation scores in one `experiments.csv`, and stores one normal run directory per policy. `--job-index` selects one zero-based matrix entry; `--total-steps` overrides the budget equally for selected jobs. `--dry-run` prints the matrix and total budget without starting training.

`agent_count.yaml` uses the full Medium task without curriculum to isolate the agent-count setting. `agent_count_curriculum.yaml` is a separate protocol with Medium curriculum. Each is evaluated on the same full final task. Model XML is generated automatically and saved self-contained in each run/checkpoint. No XML file per agent count is needed.

| Agents | Actor input | Critic input | Maximum summed actuator force per axis |
| --- | ---: | ---: | ---: |
| 2 | 26 | 25 | 6 N |
| 4 | 34 | 33 | 12 N |
| 8 | 50 | 49 | 24 N |
| 16 | 82 | 81 | 48 N |

Per-agent radius, mass and force limits and the object's physics remain fixed. More agents increase total available force/mass, collision constraints, actor samples per team step, and network input size. Improvements therefore do not isolate coordination efficiency by themselves. Report these factors, use equal team-step budgets and multiple seeds, and keep success, mechanical efficiency and coordination claims separate. The privileged scripted diagnostic remains four-agent only.

### Reward Ablation

The default remains `reward_mode: baseline`. `footprint_potential` replaces only the progress term with potential shaping: `Phi=-footprint_error` and `F=gamma*Phi(next)-Phi(current)`. Footprint error is the minimum translation needed to fit the oriented shape into the goal, with a positive deficit if its span cannot fit. It is zero exactly when the complete object fits; no target yaw or maneuver sequence is prescribed.

True terminal states (success/physics failure) have zero next potential. Time-limit truncations retain their next potential, matching GAE continuation bootstrapping. Set `shaping_gamma` equal to MAPPO `gamma`; the trainer rejects a mismatch. Success, time and failure weights remain unchanged. Raw episode returns across the two reward modes are not comparable measures of task quality; compare success and physical metrics.

```bash
python scripts/run_experiments.py configs/experiments/reward_ablation.yaml --output results/reward_ablation --dry-run
```

This declares both rewards with the same Easy curriculum, four agents, equal budgets and five seeds. The implementation has pipeline/geometry tests; learned improvements still require running the experiments.

## 6. Installation

### Prerequisites

- Python 3.10 (the tested baseline)
- An OpenGL backend for the viewer or offscreen recording
- FFmpeg with the `libx264` encoder for video recording

### Open the Project and Create a Virtual Environment

Open the repository root (the directory containing `assets/`, `configs/`, and `scripts/`) and run:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the physics/environment dependencies only: MuJoCo 3.3.7, NumPy 1.26.4, Gymnasium 1.4.0, PettingZoo 1.27.0, and PyYAML 6.0.3.

For MAPPO on CPU, also install PyTorch and the training requirements (including TensorBoard):

```bash
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
```

The training requirements pin TensorBoard 2.20.0 and a compatible setuptools version for its CLI. TensorFlow is not required for these PyTorch scalar logs.

For CUDA, use the matching PyTorch 2.6.0 build from the [official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/). TorchVision and TorchAudio are not required. Installing the CUDA build does not parallelize the single MuJoCo environment.

Install FFmpeg on Ubuntu if needed:

```bash
sudo apt install ffmpeg
```

The training CLI defaults to `MUJOCO_GL=egl` on Linux. The evaluation CLI does the same when saving videos. An explicitly supplied `MUJOCO_GL` is respected. Interactive viewing requires a working desktop display; direct headless library usage should set the backend before importing MuJoCo.

## 7. Usage

### Run Physics Diagnostics

```bash
python scripts/test_physics.py
```

The default opens the viewer for the translation scenario. Run all diagnostic scenarios without a display and save measurements with:

```bash
python scripts/test_physics.py --headless --scenario all --json results/physics_validation.json
```

### Run a Specific Physics Scenario

```bash
python scripts/test_physics.py --scenario release
```

### Run the Multi-Agent Environment

Zero-action policy:

```bash
python scripts/test_environment.py --policy zero --episodes 1
```

Random-action policy:

```bash
python scripts/test_environment.py --policy random --episodes 5
```

Scripted transport and API validation:

```bash
python scripts/test_environment.py --policy scripted --no-randomize --episodes 5 --seed 42 --require-success --json results/scripted_validation.json
python scripts/test_environment.py --policy scripted --no-randomize --render --episodes 1
python scripts/test_environment.py --api-test
```

The scripted controller is a diagnostic tool using global geometry, not the learned MAPPO policy. Without `--require-success`, a timeout is a valid episode outcome; physical failure still produces a nonzero exit code.

Run the automated tests with:

```bash
python -m unittest discover -s tests -v
```

Tests cover physics, the Parallel API, observations, rewards, GAE, PPO updates, checkpoint selection, resume, and video milestones. PyTorch-dependent tests skip if PyTorch is missing; the video integration test requires Linux EGL, FFmpeg, and ffprobe. Passing only the environment tests does not validate MAPPO.

### Inspect the Obstacle Arenas

```bash
python scripts/test_environment.py --config configs/environment_medium.yaml --policy random --render --episodes 1
python scripts/test_environment.py --config configs/environment_medium.yaml --api-test
python scripts/test_environment.py --config configs/environment_hard.yaml --policy random --render --episodes 1
python scripts/test_environment.py --config configs/environment_hard.yaml --api-test
```

`--no-randomize` gives a fixed diagnostic start in the left room with the goal in the right room. The existing scripted Easy controller does not plan a route through the obstacle courses; use random/zero actions for diagnostics or the corresponding learned policy for evaluation.

### Train MAPPO: Smoke Run

```bash
python scripts/train.py \
  --device cpu \
  --total-steps 2048 \
  --rollout-steps 256 \
  --run-dir results/mappo_training
```

This 2,048-step run checks the pipeline; it does not establish learned transport success. All examples require a new `--run-dir` so existing runs are not overwritten.

### Train MAPPO: Total Timesteps

Run the default 100,000-step baseline:

```bash
python scripts/train.py --config configs/mappo.yaml --env-config configs/environment.yaml --device cpu --run-dir results/mappo_baseline
```

For a longer GPU run with an explicit step budget and independent schedules:

```bash
python scripts/train.py --device cuda --total-steps 2000000 --eval-interval-steps 50000 --video-interval-steps 100000 --run-dir results/mappo_video
```

`--total-steps` (alias `--total-timesteps`) is the cumulative number of **team environment decisions**. One simultaneous action from all N agents counts as one timestep and produces one shared reward. It is not four timesteps or 50 physics substeps. The trainer stops at the requested budget, including a shorter final rollout when needed. Episode count remains a performance statistic; `--total-episodes` is no longer a supported training flag.

With full 500-step episodes, 100,000 timesteps complete 200 episodes. Success or physical failure can shorten episodes, so the episode count is not a fixed training budget. A step budget still does not guarantee convergence.

### Train a Separate Medium Policy

```bash
python scripts/train.py \
  --env-config configs/environment_medium.yaml \
  --device cuda \
  --total-steps 2000000 \
  --run-dir results/mappo_medium
```

Use `--device cpu` on a machine without CUDA. This starts a new policy with the existing MAPPO settings, TensorBoard logging, evaluation every 50,000 steps, 1080p video every 100,000 steps, and only `best.pt`/`latest.pt` checkpoints. Schedules are checked at rollout boundaries as described above. Two million steps is an initial experimental budget, not a success guarantee.

To resume Medium, use its own checkpoint and a new run directory. An Easy resume restores the Easy task; `--env-config` cannot switch the task of a resumed run.

```bash
python scripts/train.py --resume results/mappo_medium/checkpoints/latest.pt --device cuda --total-steps 4000000 --run-dir results/mappo_medium_resume
python scripts/evaluate.py results/mappo_medium/checkpoints/best.pt --device auto --episodes 50 --seed 200000 --json results/mappo_medium/eval_best_50.json --video-dir results/mappo_medium/videos/eval_best --video-episodes 3
```

Evaluation restores the saved Medium configuration and XML automatically. No additional difficulty argument is needed.

### Train a Separate Hard Policy

```bash
python scripts/train.py \
  --env-config configs/environment_hard.yaml \
  --device cuda \
  --total-steps 2000000 \
  --run-dir results/mappo_hard
```

Hard starts a new policy independently of Medium, with the same MAPPO defaults, TensorBoard, 1080p recording every 100,000 steps, and only `best.pt`/`latest.pt`. There are 2,500 decisions per maximum-length episode. The goal remains `(3.25, 1.5)` m across training and evaluation. Two million steps is an initial experimental budget; learned success has not been established.

```bash
python scripts/evaluate.py results/mappo_hard/checkpoints/best.pt --device auto --episodes 50 --seed 200000 --json results/mappo_hard/eval_best_50.json --video-dir results/mappo_hard/videos/eval_best --video-episodes 3
```

### Checkpoint Retention

Each new run retains only these model files:

| File | When it is replaced |
| --- | --- |
| `checkpoints/best.pt` | When evaluation improves: higher success rate first, then higher mean return |
| `checkpoints/latest.pt` | Every `checkpoint_interval_steps` team steps at a rollout boundary, at a video milestone, and on normal completion |

The default checkpoint threshold interval is 10,000 steps. Numbered update/episode checkpoints and `last.pt` are no longer created by the updated trainer. Old files in existing runs are not deleted. Checkpoint writes use a temporary file and atomic replacement.

`latest.pt` represents the most recent save, not every environment step; work collected after that save is not present if a run is interrupted.

Checkpoints contain model weights, input-normalization statistics, both optimizers, training counters/RNG, configuration, feature schema, and the MuJoCo XML/hash. Changing source configuration or XML does not change an already-running process or a checkpoint's saved physics.

**A training process started before this change still uses its old checkpoint-saving code.** Apply the new behavior by restarting and resuming into a new run directory. Old `last.pt` files remain valid resume inputs; select the actual filename that exists in the source run.

### Resume Training

From a run created with the updated trainer:

```bash
python scripts/train.py --resume results/mappo_baseline/checkpoints/latest.pt --total-steps 200000 --device cpu --run-dir results/mappo_baseline_resume
```

To continue a legacy run that used an episode budget, explicitly replace its budget with a cumulative step target:

```bash
python scripts/train.py --resume results/mappo_video/checkpoints/latest.pt --total-steps 2000000 --eval-interval-steps 50000 --video-interval-steps 100000 --device cuda --run-dir results/mappo_video_resume
```

The target must exceed the saved timestep count; it includes prior training rather than adding that many new steps. For example, resuming from 1,350,000 steps with a 2,000,000-step target collects 650,000 additional steps. A legacy episode-budget checkpoint requires an explicit `--total-steps` to avoid carrying over an unintended budget. Legacy YAML with `total_episodes` also requires an explicit step override.

Resume restores MAPPO and saved XML, while enforcing nearby randomized agents and fixed obstacle-task goals as described above. The CLI permits changing step budget, device, interval flags, `--tensorboard`/`--no-tensorboard`, and `--reward-log-interval-steps`. Do not pass `--config`, `--env-config`, `--seed`, or `--rollout-steps` to override a resumed run. Old `eval_interval`/`checkpoint_interval` values are converted from update counts to step intervals using the saved rollout length. A positive legacy `video_interval_episodes` becomes the new default 100,000-step interval (zero remains disabled); provide `--video-interval-steps` to choose another interval.

Resume starts a fresh physics episode and discards any partial rollout/episode. It restores RNG and optimizers but is not an exact continuation of uninterrupted training. Completed-episode and rolling-statistics history are restored; a discarded partial episode does not count as completed. Old checkpoints without rolling statistics start a new statistics window; old completed-episode counters may use `episode_index` as an estimate. Schedules continue at the next global step multiple, without replaying missed videos. Each resumed run selects its own `best.pt`; the previous run retains its previous best.

### Evaluate or Record a Policy Manually

```bash
python scripts/evaluate.py results/mappo_baseline/checkpoints/best.pt --episodes 10 --seed 200000 --json results/mappo_baseline/evaluation_final.json
python scripts/evaluate.py results/mappo_baseline/checkpoints/best.pt --episodes 1 --render
python scripts/evaluate.py results/mappo_baseline/checkpoints/best.pt --episodes 1 --video-dir results/mappo_baseline/videos/manual --json results/mappo_baseline/videos/manual/evaluation.json
```

The manual video is `results/mappo_baseline/videos/manual/policy_seed_200000.mp4`. Add `--stochastic` to sample actions; the default uses `tanh(mean)`. Evaluation restores the checkpoint's XML and normalization. `--env-config` explicitly selects a different environment for transfer tests. `--trajectory-dir` saves NPZ states, qpos, actions, rewards, times, and terminal flags.

The evaluation CLI prints a checkpoint-loading message, the actual device, episode starts, progress every 100 steps or about five seconds between completed steps, video initialization/finalization, and each episode's result. All progress lines flush immediately. Use `--progress-interval-steps` to change the step interval or `--quiet` to hide live progress. CPU policy inference defaults to one PyTorch thread, adjustable with `--torch-threads`.

With `--json`, the report is written atomically at startup and after each episode. During evaluation it has `complete=false`, `requested_episodes`, and `completed_episodes`; normal completion sets `complete=true`. Completed reports survive Ctrl+C, which exits with code 130. Old evaluator processes do not acquire these changes while already running.

`--video-dir` alone records **every** evaluation episode. Evaluating 50 episodes with that flag creates 50 MP4s and includes rendering/encoding time. To evaluate 50 episodes while recording only the first three:

```bash
python scripts/evaluate.py results/mappo_baseline/checkpoints/best.pt --device auto --episodes 50 --seed 200000 --json results/mappo_baseline/evaluation_50.json --video-dir results/mappo_baseline/videos/eval_best --video-episodes 3
```

During encoding, the current file has a `.partial.mp4` suffix; it is renamed after the encoder finishes. A leftover partial file alone does not establish that a process is still running. Separate video subdirectories keep these recordings distinct from training logs and other evaluations.

### Automatic Evaluation Videos

Video settings in `configs/mappo.yaml` are:

```yaml
video_interval_steps: 100000
video_width: 1920
video_height: 1080
video_crf: 18
```

At the rollout boundary reaching 100,000, 200,000, and subsequent team-step thresholds, a separate environment records one deterministic evaluation episode on seed 100000. Evaluation/video steps do not increase the training step counter. Training sampling and normalization are preserved.

Output for the first milestone:

```text
results/mappo_video/videos/step_000100352/
└── policy_seed_100000.mp4
```

The complete video report is appended to the run's `evaluations.jsonl` with `kind: "video"`; automatic recording does not create a separate JSON file in each video directory.

Videos use the top camera, native 1920 x 1080 frames, H.264/MP4, and CRF 18. Frame rate follows the control interval: 20 fps by default. Recording includes the initial state, so a 500-step episode contains 501 frames and lasts 25.05 seconds. Frames stream to FFmpeg instead of accumulating in RAM.

At a recording milestone, `latest.pt` is replaced with the updated policy being recorded. Video metadata identifies the threshold (`scheduled_step`), actual step, episode count, update, and XML hash. Its `checkpoint` path points to the mutable `latest.pt`, with `checkpoint_is_overwritten=true`: later saves replace those weights, so historical videos do not retain a separate recoverable model snapshot.

Set `--video-interval-steps 0` to disable automatic recording. Manual evaluation accepts `--video-width`, `--video-height`, and `--video-crf`; frame dimensions must be even. Startup checks the renderer/encoder when the configured budget could reach a recording milestone.

### Logs and Run Artifacts

The console prints PPO update, cumulative step budget, completed episodes, rolling mean return, success/stability rates, mean episode length, statistics-window size, rollout mean reward, steps per second (`fps`), and losses. The rolling metrics use the most recent 100 completed episodes by default. Before any episode finishes they show `n/a`, not a fabricated zero. FPS measures team decisions per wall-clock second during the current process, including updates and intervening evaluations/videos. The GPU accelerates the networks; this code still runs MuJoCo in a single CPU environment.

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Resolved settings, software versions, feature schema, device, resume source, XML hash |
| `arena.xml`, `environment.yaml`, `mappo.yaml` | Model/configuration snapshots for that run |
| `episodes.jsonl` | Team return, seed, success, stability, end reason, distance, path length, wall contacts, control effort, penetration |
| `updates.jsonl` | Losses, KL, entropy, throughput, all-transition reward/component means, rolling and rollout episode metrics |
| `evaluations.jsonl` | Complete periodic and video evaluation history, one report per line; `kind` is `evaluation` or `video` |
| `evaluation_latest.json`, `evaluation_best.json` | Two overwritten snapshots for the latest and best periodic evaluation |
| `checkpoints/best.pt`, `checkpoints/latest.pt` | The two retained training model files |
| `videos/step_XXXXXXXXX/` | MP4 indexed by actual training step; metadata lives in `evaluations.jsonl` |
| `tensorboard/events.out.tfevents.*` | Training/evaluation scalars with cumulative timesteps as the X axis |

`mean_episode_return`, `rollout_success_rate`, and `rollout_stable_rate` retain their per-rollout meaning and appear only when an episode ends in that rollout. `rolling_episode_stats` instead uses the last `stats_window_size` completed episodes and remains available across rollouts with no completed episode. Control effort is integrated squared actuator force in N²·s, not mechanical energy. Wall collision events count new contact-pair occurrences rather than every contact frame.

Training no longer creates `evaluation_XXXXXX.json` files. The full evaluation frequency and episode-level data are retained in one JSONL history; only the two convenient latest/best JSON snapshots are overwritten. Manual `scripts/evaluate.py --json ...` reports remain under the user's chosen filenames.

To compact older stopped or copied runs:

```bash
python scripts/compact_results.py results/m4_random_arena results/mappo_medium results/mappo_hard
```

The utility first archives the exact original numbered and per-video JSON bytes into one `evaluation_archive.zip` per run, verifies the ZIP, writes and verifies `evaluations.jsonl`, then removes the loose originals. Checkpoints, MP4s, TensorBoard, training logs and custom evaluation reports remain in place. Repeating the command is safe; already-consolidated records are deduplicated. Source changes during compaction abort removal. Use it after stopping a writer or on a copied run; old running processes continue creating their old output layout until restarted/resumed with the updated trainer.

Use `src.training.artifacts.read_evaluations(run_dir)` to read either layout (add `include_videos=True` for video reports). Original filenames and bytes remain recoverable from the ZIP archive if needed.

### TensorBoard Monitoring

TensorBoard is enabled by default and installed through `requirements-training.txt`. Open it while training runs:

```bash
tensorboard --logdir results --port 6006
```

Visit `http://localhost:6006` on that machine. For an SSH remote, run this on your local computer and keep the tunnel open:

```bash
ssh -L 6006:localhost:6006 USER@REMOTE_HOST
```

The TensorBoard process runs on the remote PC; your local browser uses `http://localhost:6006`. If another process uses port 6006, choose matching alternative ports.

| Scalar tags | Meaning |
| --- | --- |
| `rollout/ep_rew_mean`, `rollout/ep_len_mean` | Mean episode return/length over the recent completed-episode window |
| `rollout/success_rate`, `rollout/stable_rate` | Recent training fractions in [0, 1] |
| `rollout/window_episodes` | Actual number of completed episodes used in that window |
| `episode/return`, `episode/length`, `episode/success`, `episode/stable` | Individual completed-episode values, located at their terminal timestep |
| `reward/step` | Raw shared reward sampled every `reward_log_interval_steps` (default 100) |
| `reward/mean_step` | Mean reward across every transition in the rollout |
| `reward/progress_mean`, `reward/time_mean`, `reward/success_mean`, `reward/failure_mean` | Mean weighted reward components across all rollout transitions; sum to `reward/mean_step` within rounding |
| `eval/mean_return`, `eval/success_rate`, `eval/stable_rate` | Separate deterministic validation results |
| `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/approx_kl` | PPO update diagnostics |
| `time/fps`, `time/total_timesteps`, `time/episodes` | Throughput and cumulative progress |

Reward is computed on **every** environment step. Default raw sampling reduces event-file size and may miss isolated bonuses; rollout component means include them all. Set `--reward-log-interval-steps 1` to graph every raw step reward, or 0 to omit raw samples. Episode return is a sum of rewards; it is not the same quantity as mean reward per step. Rolling reward metrics describe stochastic training, while `eval/` metrics describe deterministic evaluation.

Use `--no-tensorboard` to keep JSON/console logging without writing event files. Resume writes a new run directory but continues scalar X coordinates from the saved timestep count. Existing JSON-only runs are preserved; their historical points are not automatically imported into TensorBoard. The integration uses [PyTorch SummaryWriter](https://docs.pytorch.org/docs/2.6/tensorboard.html).

Low policy/value loss alone does not demonstrate successful transport. Compare return, success, stability, motion, and videos. Default evaluation reuses five validation seeds (100000-100004); this is not a generalization benchmark or five independent training seeds. Deterministic evaluation and stochastic training can have different success rates.

Local `results/`, logs, and checkpoint files are listed in `.gitignore` and are not bundled with a fresh checkout. Create artifacts by running the commands above.

## 8. Research Objectives

This project is designed to investigate cooperative behavior in physically interacting multi-agent systems.

The main research questions include:

- Can multiple independent agents learn to transport a shared object through reinforcement learning?
- How do cooperative pushing strategies emerge from shared rewards?
- How do agents coordinate translation and rotation of an asymmetric object?
- How does the number of participating agents influence transport performance?
- How do observation availability and environmental constraints affect cooperation?
- Can learned cooperative strategies generalize to different object configurations and obstacle layouts?

Limited-range sensors, communication, randomized obstacle layouts, energy/collision reward ablations, and held-out layouts require additional implementation. Agent-count matrices, staged reset curricula and the footprint-potential reward ablation are implemented. The fixed Medium and Hard environments are implemented; learned success on them still needs to be demonstrated. Systematic results require multiple independent training seeds and an explicit evaluation protocol.

## 9. References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

**Cooperative T-Shape Transport — Multi-Agent Reinforcement Learning with MuJoCo and MAPPO.**
