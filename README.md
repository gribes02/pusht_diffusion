# Diffusion Policy for PushT

A from-scratch implementation of **Diffusion Policy** applied to the **PushT** robotic manipulation task. The goal is to train a robot agent to push a T-shaped block to a target pose using a denoising diffusion probabilistic model (DDPM) as the policy.

> ⚠️ This project is currently under active development. Not all components are complete.

---

## What is Diffusion Policy?

Diffusion Policy ([Chi et al., 2023](https://diffusion-policy.cs.columbia.edu/)) is a robot learning method that frames action generation as a denoising process. Instead of directly predicting an action, the policy learns to iteratively denoise a sequence of randomly sampled Gaussian noise into a clean action trajectory, conditioned on the current observation.

This makes it particularly good at modelling **multimodal action distributions** — situations where there are multiple valid ways to accomplish a task — which traditional behaviour cloning methods struggle with.

---

## The Task: PushT

PushT is a 2D robotic manipulation environment where a robot end-effector must push a T-shaped block to a target position and orientation. It is a standard benchmark for imitation learning because it requires precise, contact-rich control and has inherent multimodality (the block can be approached from multiple sides).

- **Observation**: end-effector position + T-block pose (position + angle)
- **Action**: 2D end-effector velocity / target position
- **Success**: T-block overlaps with the target pose above a threshold

---

## How It Works

### Training

1. **Collect demonstrations** — A dataset of expert trajectories is collected, each consisting of sequences of observations and actions.
2. **Normalise** — Observations and actions are normalised to `[-1, 1]` using dataset statistics.
3. **Add noise (forward process)** — For each training sample, a random timestep $t$ is sampled and Gaussian noise is added to the clean action sequence according to the DDPM forward process:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

4. **Predict noise (reverse process)** — A U-Net is trained to predict the noise $\epsilon$ that was added, conditioned on the observation encoding and the timestep $t$:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, o) \|^2 \right]$$

5. **EMA** — An exponential moving average (EMA) of the model weights is maintained during training for more stable inference.

### Inference

At inference time, the policy generates an action sequence from pure noise by running the full DDPM reverse chain for $T$ steps (e.g. $T=100$), conditioned on the current observation:

$$x_{t-1} = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t} \hat{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \sigma_t z$$

where $\hat{x}_0$ is the clean action reconstructed from the model's noise prediction, and $z \sim \mathcal{N}(0, I)$.

Only a portion of the predicted action sequence (the **action horizon**) is executed in the environment before re-planning.

---

## Project Structure

```
pusht_diffusion/
│
├── configs/
│   └── pusht_default.yaml          # Hyperparameters: obs_dim, action_dim, horizons, diffusion steps, etc.
│
├── diffusion/
│   ├── noise_scheduler.py          # DDPM forward (add_noise) and reverse (step) process  ✅
│   ├── unet.py                     # Conditional U-Net: the noise prediction network       🚧
│   ├── obs_encoder.py              # Encodes observations into a conditioning vector       🚧
│   └── diffusion_policy.py         # Wraps encoder + U-Net + scheduler into a full policy 🚧
│
├── training/
│   ├── trajectory_dataset.py       # PyTorch Dataset for loading expert demonstrations    🚧
│   ├── normalizer.py               # Normalises obs and actions to [-1, 1]                🚧
│   └── trainer.py                  # Training loop with EMA, logging, checkpointing       🚧
│
├── environment/
│   └── pusht_env.py                # Thin wrapper around the PushT gymnasium environment  ✅
│
├── utils/
│   └── ema.py                      # Exponential Moving Average of model weights          🚧
│
├── train.py                        # Entry point for training                             🚧
├── eval.py                         # Entry point for evaluation / rollout                 🚧
└── requirements.txt                # Python dependencies
```

**Legend:** ✅ Implemented &nbsp;|&nbsp; 🚧 In progress

---

## Components In Detail

### `DDPMScheduler` (`diffusion/noise_scheduler.py`)
Implements the DDPM noise schedule. Supports `linear` and `quadratic` beta schedules.
- `add_noise(x0, noise, timesteps)` — forward process, adds noise to a batch of clean actions
- `step(model_output, timestep, sample)` — single reverse denoising step

### U-Net (`diffusion/unet.py`)
A 1D temporal U-Net that takes a noisy action sequence, a timestep embedding, and an observation conditioning vector, and outputs the predicted noise. Will include:
- Sinusoidal timestep embeddings
- Residual convolutional blocks with FiLM conditioning
- Skip connections between encoder and decoder

### Observation Encoder (`diffusion/obs_encoder.py`)
Encodes the observation history (a window of past observations) into a fixed-size conditioning vector that is passed to the U-Net. Will be an MLP for state-based observations.

### Diffusion Policy (`diffusion/diffusion_policy.py`)
The top-level policy class combining the encoder, U-Net, and scheduler. Exposes:
- `predict_noise(noisy_actions, timesteps, obs)` — single forward pass during training
- `generate_actions(obs)` — full denoising loop at inference time

### Dataset & Normaliser (`training/`)
- `TrajectoryDataset` — loads demonstration trajectories, chunks them into overlapping windows of `(obs_horizon, action_horizon)` pairs
- `Normalizer` — fits min/max statistics from the dataset and normalises/unnormalises tensors

### Trainer (`training/trainer.py`)
The main training loop:
1. Sample a batch of `(obs, action)` windows
2. Sample random timesteps and add noise to actions
3. Forward pass through the U-Net
4. Compute MSE loss against the true noise
5. Backpropagate and update weights + EMA model

### EMA (`utils/ema.py`)
Maintains a shadow copy of model weights updated as:

$$\theta_{\text{ema}} \leftarrow \tau \cdot \theta_{\text{ema}} + (1 - \tau) \cdot \theta$$

The EMA model is used for inference only.

---

## Installation

```bash
git clone https://github.com/your-username/pusht_diffusion.git
cd pusht_diffusion
pip install -r requirements.txt
pip install pusht  # PushT gymnasium environment
```

---

## Usage

### Training
```bash
python train.py --config configs/pusht_default.yaml
```

### Evaluation
```bash
python eval.py --checkpoint checkpoints/latest.pth
```

---

## Configuration

Key hyperparameters in `configs/pusht_default.yaml`:

| Parameter | Description |
|---|---|
| `obs_dim` | Observation dimension |
| `action_dim` | Action dimension (2 for PushT) |
| `obs_horizon` | Number of past observations fed to the policy |
| `action_horizon` | Number of future actions to execute per planning step |
| `pred_horizon` | Number of future actions predicted per denoising pass |
| `num_train_timesteps` | Number of DDPM diffusion steps (e.g. 1000) |
| `num_inference_timesteps` | Denoising steps at inference (e.g. 100) |
| `beta_schedule` | `linear` or `quadratic` |

---

## References

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/) — Chi et al., 2023
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., 2020
- [PushT Environment](https://github.com/huggingface/gym-pusht)
