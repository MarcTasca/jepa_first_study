# Non-Contrastive Learning of Physical Dynamics (JEPA)

A PyTorch implementation of **Joint Embedding Predictive Architecture (JEPA)** designed to learn continuous physical dynamics directly from high-dimensional observations (pixels), without relying on reconstruction-based losses in the representation learning phase. This project applies Self-Supervised Learning (SSL) to capture the underlying differential equations of a chaotic double pendulum.

## Abstract

Self-supervised learning has shown remarkable success in verifying semantic content in static images (I-JEPA) and videos (V-JEPA). This work extends these principles to **continuous physical systems**, specifically targeting the challenge of long-term autoregressive forecasting in latent space. By combining **Multistep VICReg Regularization** with **Residual Predictive Dynamics**, we enforce a representation that is not only informative but also temporally consistent and stable, effectively learning a "Neural ODE" of the system's manifold.

---

## Methodology

### 1. Joint Embedding Predictive Architecture (JEPA)
The core architecture consists of three components trained to minimize prediction error in latent space:

*   **Context Encoder ($E_\theta$):** Maps a short history of frames ($x_{t-k} \dots x_t$) to a latent state $z_t$.
*   **Target Encoder ($E_\phi$):** An Exponential Moving Average (EMA) of the Context Encoder. It provides stable regression targets ($z_{t+k}$) for future states.
    *   Update rule: $\phi_t \leftarrow \mu \phi_{t-1} + (1 - \mu) \theta_t$
*   **Residual Predictor ($P_\psi$):** A wide MLP that models the *state transition* rather than the state itself.
    *   Formulation: $z_{t+1} = z_t + P_\psi(z_t)$
    *   Rationale: This residual formulation mimics the structure of Euler integration in ODEs ($s_{t+1} = s_t + \Delta t \cdot f(s_t)$), preventing gradient explosion and drift during long-horizon unrolling.

### 2. Multistep VICReg (Trajectory Consistency)
Unlike standard VICReg which operates on static multiview pairs, we introduce a **Multistep** variant tailored for dynamics. We unroll the predictor for $H$ steps and apply regularization at every step of the trajectory.

The total loss $\mathcal{L}$ is a weighted sum of three components averaged over the prediction horizon $H$:

$$ \mathcal{L} = \frac{1}{H} \sum_{k=1}^{H} \left( \lambda \mathcal{L}_{inv}^{(k)} + \mu \mathcal{L}_{var}^{(k)} + \nu \mathcal{L}_{cov}^{(k)} \right) $$

1.  **Invariance ($\mathcal{L}_{inv}$):** Standard MSE between predicted state $\hat{z}_{t+k}$ and target state $z_{t+k}$.
2.  **Variance ($\mathcal{L}_{var}$):** Hinge loss enforcing the standard deviation of each embedding dimension to be at least 1, preventing collapse to a single point.
3.  **Covariance ($\mathcal{L}_{cov}$):** Penalizes off-diagonal elements of the embedding covariance matrix, forcing dimensions to be decorrelated and maximizing information content.

### 3. Decoupled Decoder Training
To strictly verify the quality of the learned representation, we train a Decoder **after** the JEPA phase is complete. The Encoder is frozen (`requires_grad=False`), and the Decoder attempts to reconstruct pixels $x_t$ from $z_t$. High reconstruction quality confirms that the self-supervised phase captured the necessary physical state information.

---

## System Architecture

| Component | Specification | Rationale |
| :--- | :--- | :--- |
| **Vision Encoder** | CNN (64 $\to$ 128 $\to$ 256 ch) | High-capacity feature extraction to resolve fine details (e.g., the second arm of the pendulum). |
| **Vision Decoder** | Transpose Conv (256 $\to$ 128 $\to$ 64 ch) | Symmetric capacity to ensure pixel-perfect reconstruction during verification. |
| **Latent Dim** | 64 | Tighter abstraction than pixel space ($64 \ll 64 \times 64 \times 3$), forcing semantic compression. |
| **Predictor** | Residual MLP (512 width) | Models the velocity field ($\frac{dz}{dt}$) of the latent manifold. |
| **Input History** | 2 Frames | Minimal temporal context required to infer velocity from static positions. |

---

## Technical Implementation

### Training Curriculum
1.  **Phase I: Dynamics Learning (JEPA)**
    *   **Objective:** Minimize Multistep VICReg loss.
    *   **Mechanism:** Unrolls predictions for 8 steps (Horizon). Backpropagates through time (BPTT).
    *   **Scheduler:** `ReduceLROnPlateau` (Patience=5) to fine-tune convergence.
2.  **Phase II: Grounding (Decoder)**
    *   **Objective:** Minimize Pixel MSE ($|| \hat{x} - x ||^2$).
    *   **Constraint:** Encoder is frozen.
    *   **Result:** Proves that $z_t$ maps isomorphically to the physical state.

### Codebase Structure
*   `src/models.py`: Definitions of the Scaled Vision Encoder/Decoder and Residual Predictor.
*   `src/trainer.py`: Implementation of the dual-phase training loop and Multistep VICReg loss.
*   `src/runner.py`: Experiment orchestration and logging.
*   `src/dataset.py`: On-the-fly Runge-Kutta 4 (RK4) physics rendering and data caching.

---

## Usage

### Prerequisites
*   Python 3.10+
*   `uv` (Universal Package Manager) or `pip`

### Development Commands

```bash
# Run Unit Tests
uv run python -m pytest

# Run Linter & Formatter
uv run pre-commit run --all-files
```

### Training

To reproduce the full experiment on the Double Pendulum dataset:

```bash
uv run python run.py \
  --mode pendulum_image \
  --size 10000 \
  --image_size 64 \
  --epochs 50 \
  --decoder_epochs 100 \
  --batch_size 256
```

### Key Arguments
*   `--mode`: Selects the dataset (`pendulum_image` for pixels, `pendulum` for raw coordinates).
*   `--prediction_horizon`: Number of autoregressive steps to unroll during training (Default: 8).
*   `--batch_size`: Larger batch sizes (e.g., 256) provide better statistics for the VICReg Covariance loss.
*   `--decoder_epochs`: Duration of the verification phase (Phase II).

---

## References

**[1] I-JEPA**
Assran, M., et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture". *CVPR 2023*. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

**[2] V-JEPA**
Bardes, A., et al. (2024). "Revisiting Feature Prediction for Learning Visual Representations from Video". *TMLR 2024*. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)

**[3] VICReg**
Bardes, A., Ponce, J., & LeCun, Y. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning". *ICLR 2022*. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)

**[4] Neural ODEs**
Chen, R. T. Q., et al. (2018). "Neural Ordinary Differential Equations". *NeurIPS 2018*. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

**[5] Barlow Twins**
Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). "Barlow Twins: Self-Supervised Learning via Redundancy Reduction". *ICML 2021*. [arXiv:2103.03230](https://arxiv.org/abs/2103.03230)
