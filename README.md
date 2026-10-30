# Cooperative T-Shape Transport Using Multi-Agent Reinforcement Learning

**A MuJoCo-based multi-agent reinforcement learning framework for cooperative object transport using Multi-Agent Proximal Policy Optimization (MAPPO).**

This project explores how multiple autonomous agents can learn to coordinate their actions to transport a shared rigid object toward a target location.

Inspired by cooperative geometric puzzle-solving experiments involving ants and humans, the project adapts collective object transportation into a physics-based multi-agent reinforcement learning environment.

The primary objective is to investigate **emergent cooperative behavior**, where multiple agents learn to coordinate through physical interactions without explicitly programmed individual roles.

The current implementation is an **Easy arena baseline with four agents and global actor information**. Obstacle courses, limited-range observations, variable agent counts, and generalization benchmarks are research extensions. Successful transport alone does not establish emergent coordination.

---

## 1. Project Overview

The simulation consists of four circular robotic agents interacting with a passive rigid object inside a bounded arena.

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
- Automatic 1080p evaluation videos every 10,000 completed training episodes
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

The environment contains four circular robotic agents.

Each agent is independently controlled through two force actuators corresponding to the X and Y directions.

| Parameter              | Value         |
| ---------------------- | ------------- |
| Number of agents       | 4             |
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

The arena interior measures **5.8 x 5.8 m**, with boundary walls and no internal obstacles. Bodies remain at fixed height with zero gravity; the floor is visual only. Support resistance is approximated by joint damping and dry friction rather than physical floor contact.

### Physics Configuration

| Parameter                     | Value              |
| ----------------------------- | ------------------ |
| Physics engine                | MuJoCo             |
| Physics timestep              | 0.001 s            |
| Physics substeps per action   | 50                 |
| Control interval              | 0.05 s             |
| Control frequency             | 20 Hz              |
| Maximum episode duration      | 25 s               |
| Maximum environment steps     | 500                |
| Contact response              | `solref="0.002 1"` |
| Contact penetration threshold | 5 mm               |

The simulation uses a smaller physics timestep to improve contact resolution while maintaining a consistent control frequency.

The object uses XY damping of 4 N·s/m and yaw damping of 0.8 N·m·s/rad, with dry friction of 1.5 N per translation axis and 0.16 N·m on yaw. Agent damping is 1 N·s/m per axis. Geometry, mass, damping, actuator limits, and contact settings are defined in [arena.xml](assets/mujoco/arena.xml). Task settings are in [environment.yaml](configs/environment.yaml).

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

Each actor receives **34 float32 features**. Each agent has its own observation vector, with relative positions measured from that agent and vectors expressed along world XY axes:

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

The centralized critic receives a separate **33-feature global state** containing object motion, goal geometry, all agent positions/velocities, arena bounds, and episode fractions. Feature order is defined in [observations.py](src/envs/observations.py).

Actor execution does not call the critic. However, actor information is currently global and has no sensor-range masking. Limited local perception and communication experiments are not implemented yet.

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

All four agents receive the same team reward. Training stores it once per team transition; it is not multiplied by the agent count. Episode return is the sum of these rewards over an episode. A stationary 500-step episode has a return near -0.5; moving 1 cm closer in one ordinary step gives 0.019.

There is currently no reward for an agent merely approaching/touching the object, no energy penalty, and no ordinary collision penalty. Valid pushing contact is necessary. Wall contacts and control effort are logged as diagnostics; a contact triggers the failure penalty only if it leads to a physical safety violation. See [rewards.py](src/envs/rewards.py) for the implementation.

### Episode Termination

An episode ends when:

1. **Success (`terminated=True`):** all corners of all three object geometries fit inside the goal rectangle, linear speed is at most 0.05 m/s, and absolute yaw rate is at most 0.1 rad/s for five consecutive decisions (0.25 s by default).
2. **Time limit (`truncated=True`):** 500 decisions are reached without success or physical failure.
3. **Physical failure (`terminated=True`):** excessive penetration, leaving the arena, non-finite state, MuJoCo warnings, excessive speed, or an unexpected automatic simulator reset is detected.

The goal center is `(1.8, 1.5)` m with half-size `(0.75, 0.65)` m. There is no single required yaw: any orientation that fits and satisfies the settling criteria is allowed. Diagnostics run after every physics substep. The linear speed threshold of 5 m/s and object yaw-rate threshold of 10 rad/s are failure checks, not velocity controllers.

Reset can randomize object XY position by +/-0.1 m, yaw by +/-0.15 rad, and agent spawn positions by +/-0.02 m. Overlapping/out-of-bounds spawns are rejected. Supplying the same seed reproduces reset and action-space sampling within the same software/model configuration.

## 5. MAPPO Algorithm

This project implements **Multi-Agent Proximal Policy Optimization (MAPPO)** for cooperative continuous control.

This implementation combines PPO's clipped objective with a shared actor and centralized team-value estimation.

### Shared Actor

All agents share the same actor network parameters.

The actor receives observations and produces a probability distribution over continuous actions.

A Gaussian policy is used to sample actions, followed by a `tanh` transformation to keep actions within the allowed range.

The network is `34 -> 128 -> 128 -> 2`, with Tanh hidden activations and a learned, state-independent log standard deviation. The actor stores pre-tanh samples and includes the transformation's log-Jacobian in action likelihoods.

### Centralized Critic

The centralized critic estimates the expected cumulative reward using global state information.

This allows the critic to evaluate the collective state of the environment during training.

The critic is `33 -> 256 -> 256 -> 1`, with Tanh hidden activations and one scalar value per team state.

### Generalized Advantage Estimation

GAE estimates the advantage of selected actions. Let $T_t$ indicate true termination and $U_t$ indicate time-limit truncation. The implementation uses separate masks for bootstrapping and carrying advantages:

$$
\delta_t = r_t + \gamma(1-T_t)V_{\phi}(s_{t+1}) - V_{\phi}(s_t),
$$

$$
\hat{A}_t = \delta_t + \gamma\lambda(1-T_t)(1-U_t)\hat{A}_{t+1}.
$$

True termination does not bootstrap. A time limit bootstraps from the final state **before reset**, while advantage accumulation stops at either kind of episode boundary. A rollout cut also uses the next-state value. The same team advantage is used for the four actor samples at that time step. See [gae.py](src/algorithms/gae.py).

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
| Evaluation interval / episodes | Every 10 PPO updates / 5 episodes |
| Latest-checkpoint interval | Every 10 PPO updates, at video milestones, and on normal completion |
| Video interval | Every 10,000 completed training episodes |

A rollout of 1,024 team decisions provides 4,096 actor samples and 1,024 team-value samples. Training currently uses one environment process.

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

For MAPPO on CPU, also install PyTorch and the training requirements:

```bash
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
```

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
python scripts/test_environment.py --policy scripted --episodes 5 --seed 42 --require-success --json results/scripted_validation.json
python scripts/test_environment.py --policy scripted --no-randomize --render --episodes 1
python scripts/test_environment.py --api-test
```

The scripted controller is a diagnostic tool using global geometry, not the learned MAPPO policy. Without `--require-success`, a timeout is a valid episode outcome; physical failure still produces a nonzero exit code.

Run the automated tests with:

```bash
python -m unittest discover -s tests -v
```

Tests cover physics, the Parallel API, observations, rewards, GAE, PPO updates, checkpoint selection, resume, and video milestones. PyTorch-dependent tests skip if PyTorch is missing; the video integration test requires Linux EGL, FFmpeg, and ffprobe. Passing only the environment tests does not validate MAPPO.

### Train MAPPO: Smoke Run

```bash
python scripts/train.py \
  --device cpu \
  --total-steps 2048 \
  --rollout-steps 256 \
  --run-dir results/mappo_training
```

This 2,048-step run checks the pipeline; it does not establish learned transport success. All examples require a new `--run-dir` so existing runs are not overwritten.

### Train MAPPO: Baseline and Episode Budget

Run the default 100,000-step baseline:

```bash
python scripts/train.py --config configs/mappo.yaml --env-config configs/environment.yaml --device cpu --run-dir results/mappo_baseline
```

For a budget of 20,000 completed episodes and automatic videos every 10,000 episodes:

```bash
python scripts/train.py --device cpu --total-episodes 20000 --video-interval-episodes 10000 --run-dir results/mappo_video
```

`--total-episodes` takes precedence over YAML `total_steps`. The CLI accepts only one of `--total-steps` and `--total-episodes`; an explicit `--total-steps` clears a YAML episode budget. Budgets are cumulative on resume.

**Episodes and steps are different counters.** With 500 decisions per episode, 100,000 steps complete 200 episodes, and 10,000 full-length episodes require 5,000,000 steps. Episodes ending early require fewer steps. Training budget does not guarantee convergence.

### Checkpoint Retention

Each new run retains only these model files:

| File | When it is replaced |
| --- | --- |
| `checkpoints/best.pt` | When evaluation improves: higher success rate first, then higher mean return |
| `checkpoints/latest.pt` | Every `checkpoint_interval` PPO updates, at a video milestone, and on normal completion |

The default checkpoint interval is 10 updates. Numbered update/episode checkpoints and `last.pt` are no longer created by the updated trainer. Old files in existing runs are not deleted. Checkpoint writes use a temporary file and atomic replacement.

`latest.pt` represents the most recent save, not every environment step; work collected after that save is not present if a run is interrupted.

Checkpoints contain model weights, input-normalization statistics, both optimizers, training counters/RNG, configuration, feature schema, and the MuJoCo XML/hash. Changing source configuration or XML does not change an already-running process or a checkpoint's saved physics.

**A training process started before this change still uses its old checkpoint-saving code.** Apply the new behavior by restarting and resuming into a new run directory. Old `last.pt` files remain valid resume inputs; select the actual filename that exists in the source run.

### Resume Training

From a run created with the updated trainer:

```bash
python scripts/train.py --resume results/mappo_baseline/checkpoints/latest.pt --total-steps 200000 --device cpu --run-dir results/mappo_baseline_resume
```

For an episode-budget run:

```bash
python scripts/train.py --resume results/mappo_video/checkpoints/latest.pt --total-episodes 30000 --device cpu --run-dir results/mappo_video_resume
```

The target must exceed the saved budget counter. Resume restores MAPPO/environment settings; the CLI permits changing budget, device, and video interval. Do not pass `--config`, `--env-config`, `--seed`, or `--rollout-steps` to override a resumed run.

Resume starts a fresh physics episode and discards any partial rollout/episode. It restores RNG and optimizers but is not an exact continuation of uninterrupted training. Completed-episode/video counters are restored; a discarded partial episode does not count as completed. Older checkpoints without this counter use `episode_index` as an estimate. Each resumed run selects its own `best.pt`; the previous run retains its previous best.

### Evaluate or Record a Policy Manually

```bash
python scripts/evaluate.py results/mappo_baseline/checkpoints/best.pt --episodes 10 --seed 200000 --json results/mappo_baseline/evaluation_final.json
python scripts/evaluate.py results/mappo_baseline/checkpoints/best.pt --episodes 1 --render
python scripts/evaluate.py results/mappo_baseline/checkpoints/best.pt --episodes 1 --video-dir results/mappo_baseline/videos/manual --json results/mappo_baseline/videos/manual/evaluation.json
```

The manual video is `results/mappo_baseline/videos/manual/policy_seed_200000.mp4`. Add `--stochastic` to sample actions; the default uses `tanh(mean)`. Evaluation restores the checkpoint's XML and normalization. `--env-config` explicitly selects a different environment for transfer tests. `--trajectory-dir` saves NPZ states, qpos, actions, rewards, times, and terminal flags.

### Automatic Evaluation Videos

Video settings in `configs/mappo.yaml` are:

```yaml
video_interval_episodes: 10000
video_width: 1920
video_height: 1080
video_crf: 18
```

At 10,000, 20,000, and subsequent completed training episodes, a separate environment records one deterministic evaluation episode on seed 100000. Successes, timeouts, and physical failures each count once toward the training counter; video evaluations do not count. Sampling/normalization during training is preserved.

Output for the first milestone:

```text
results/mappo_video/videos/episode_010000/
├── policy_seed_100000.mp4
└── evaluation.json
```

Videos use the top camera, native 1920 x 1080 frames, H.264/MP4, and CRF 18. Frame rate follows the control interval: 20 fps by default. Recording includes the initial state, so a 500-step episode contains 501 frames and lasts 25.05 seconds. Frames stream to FFmpeg instead of accumulating in RAM.

At a recording milestone, `latest.pt` is replaced with the policy being recorded, before the next PPO update. Video metadata identifies its episode, step, update, and XML hash. Its `checkpoint` path points to the mutable `latest.pt`, with `checkpoint_is_overwritten=true`: later saves replace those weights, so historical videos do not retain a separate recoverable model snapshot.

Set `--video-interval-episodes 0` to disable automatic recording. Manual evaluation accepts `--video-width`, `--video-height`, and `--video-crf`; frame dimensions must be even. Startup checks the renderer/encoder when the configured budget could reach a recording milestone.

### Logs and Run Artifacts

The console prints PPO update, budget progress, policy/value loss, and `episodes`, which counts episodes ending within that rollout. For episode budgets, `episodes_total` shows cumulative completed episodes. Evaluation prints success rate and mean return every 10 updates by default and on normal completion. Richer training metrics are saved to disk even though the console does not print them all.

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Resolved settings, software versions, feature schema, device, resume source, XML hash |
| `arena.xml`, `environment.yaml`, `mappo.yaml` | Model/configuration snapshots for that run |
| `episodes.jsonl` | Team return, seed, success, stability, end reason, distance, path length, wall contacts, control effort, penetration |
| `updates.jsonl` | Losses, entropy, KL, clipping, gradients, explained variance, sample counts, and episode metrics |
| `evaluation_XXXXXX.json` | Deterministic evaluation indexed by PPO update |
| `checkpoints/best.pt`, `checkpoints/latest.pt` | The two retained training model files |
| `videos/episode_XXXXXX/` | Milestone video and its evaluation report |

`mean_episode_return`, `rollout_success_rate`, and `rollout_stable_rate` appear in update logs only when at least one episode ends in that rollout. `episodes=0` does not mean training stopped. Control effort is integrated squared actuator force in N²·s, not mechanical energy. Wall collision events count new contact-pair occurrences rather than every contact frame.

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

Experiments with other agent counts, limited-range sensors, communication, internal obstacles/curricula, energy/collision reward ablations, and held-out layouts require additional implementation. They should not be reported as supported or solved by the current four-agent Easy baseline. Systematic results require multiple independent training seeds and an explicit evaluation protocol.

## 9. References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

**Cooperative T-Shape Transport — Multi-Agent Reinforcement Learning with MuJoCo and MAPPO.**
