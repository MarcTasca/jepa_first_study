# JEPA for Visual Dynamics (I-JEPA Implementation)

A PyTorch implementation of **Joint Embedding Predictive Architecture (JEPA)** applied to physical dynamics (Double Pendulum). This project explores how self-supervised learning can acquire a "world model" of physics directly from pixels, without reconstruction loss in the representation space.

## 🧠 Core Strategy

The model creates a compressed "mental representation" (latent space) of the world and learns to predict how that representation evolves over time.

### 1. The Components
*   **Context Encoder (`VisionEncoder`):** Maps a stack of 2 past frames ($x_{t}, x_{t-1}$) to a latent state $z_t$.
*   **Target Encoder (EMA Teacher):** A slowly moving copy of the encoder. It maps future frames ($x_{t+k}$) to targets $z_{t+k}$. It provides stable targets for learning.
*   **Predictor (Residual MLP):** Learns the physics. Given $z_t$, it predicts $z_{t+k}$.
    *   *Residual Architecture:* $s_{next} = s_{curr} + \text{Net}(s_{curr})$. This forces the network to learn the *updates* (velocity) rather than absolute states, preventing drift.
*   **Decoder (`VisionDecoder`):** Maps latents back to pixels. *Used only for verification*, not for JEPA training.

### 2. The Training Recipe
The system is trained in two decoupled phases. The separation ensures that the representation is learned purely from dynamics (Phase 1), not from pixel reconstruction.

#### Phase 1: JEPA (Self-Supervised Dynamics)
*Goal: Learn a latent space where dynamics are linear/predictable.*

1.  **Stop-Gradient Target:** We feed future frames $x_{t+k}$ into the **Target Encoder** to get $z_{target}$. Crucially, **gradients are not propagated** through the Target Encoder.
2.  **EMA Update:** The Target Encoder weights ($\phi$) are updated as an exponential moving average of the Context Encoder weights ($\theta$):
    $$ \phi_t = \mu \phi_{t-1} + (1 - \mu) \theta_t $$
    *   $\mu$ increases from 0.99 to 1.0 during training (Cosine Schedule).
3.  **Forward Pass:**
    *   Context $x_t \to$ Encoder $\to z_t$.
    *   $z_t \to$ Predictor $\to \hat{z}_{t+k}$ (Residual prediction).
4.  **Loss Calculation:**
    $$ L = \underbrace{|| \hat{z}_{t+k} - z_{target} ||^2}_{\text{MSE: Match Dynamics}} + \underbrace{0.1 \cdot \text{ReLU}(1.0 - \sigma(z))}_{\text{Reg: Prevent Collapse}} $$

#### Phase 2: Decoder (Verification / Grounding)
*Goal: Verify what information is actually captured in the latent space.*

1.  **Frozen Encoder:** The `VisionEncoder` weights are **locked** (`requires_grad=False`). We strictly evaluate the representation learned in Phase 1.
2.  **Supervised Reconstruction:**
    *   Input: Frozen latent $z_t$ from Phase 1.
    *   Target: Original pixels $x_t$.
    *   Network: `VisionDecoder` (MLP or CNN Transpose).
3.  **Optimization:** Standard MSE Loss between predicted pixels and actual pixels.
    $$ L_{dec} = || \text{Decoder}(z_t) - x_t ||^2 $$
    *   *Note:* If $L_{dec}$ is low, it proves $z_t$ contains shape/position info. If $L_{dec}$ is high, the encoder failed to capture the physical state.

---

## 🏗 Architecture Details

| Component | Specification | Reason |
|-----------|---------------|--------|
| **Latent Dim** | 64 | Enough for Double Pendulum physics (4 vars), tight enough to force abstraction. |
| **Hidden Dim** | 512 | Wide MLP to model complex chaotic dynamics. |
| **Predictor** | Residual | Essential for stable long-term autoregression (prevents exploding gradients). |
| **Input** | 2 Frames | Captures velocity (position alone is ambiguous). |

---

## 🚀 Usage

### 1. Quick Start
Train the model on the chaotic Double Pendulum dataset:

```bash
uv run python run.py \
  --mode pendulum_image \
  --epochs 50 \
  --decoder_epochs 100 \
  --batch_size 128 \
  --image_size 64
```

### 2. Key Arguments
*   `--mode`: `pendulum_image` (pixels) or `pendulum` (coordinates).
*   `--epochs`: JEPA training epochs (Phase 1).
*   `--decoder_epochs`: Independent decoder training (Phase 2).
*   `--size`: Dataset size (default 10000).

### 3. Output
Artifacts are saved to `results/<mode>_<timestamp>/`:
*   `forecast.mp4`: Side-by-side video of Ground Truth, Forecast, and Reconstruction.
*   `reconstruction.png`: Static comparison.
*   `models/*.pth`: Saved weights.

---

## 🔬 Known Phenomena

### Representation Collapse
Without regularization, JEPA reduces to a constant output ($z=0$) because that perfectly satisfies $\text{MSE}(0, 0) = 0$.
**Our Solution:** A lightweight variance penalty ($\lambda=0.1$) pushes the embeddings apart just enough to be useful, without distorting the manifold.

### Forecasting "Blindness"
If the Predictor is weak, it may only learn "identity" ($z_{t+1} \approx z_t$).
**Our Solution:** Residual connections ($z + \Delta$) and high capacity (512 width) enable learning the subtle gradients of chaotic motion.

---

---

## ⚖️ Design Choices

### Why MSE for Image Decoding?
We use Mean Squared Error (MSE) for the decoder. While MSE often causes "blurry" results in natural image generation (due to averaging multiple modes), it works well here because:
1.  **Deterministic Geometry:** The mapping from physical state (angles) to pixels is rigid. There is no ambiguity like in texture generation.
2.  **Gradient Flow:** MSE provides strong, smooth gradients for position errors. If a pendulum arm is offset by 2 pixels, MSE pulls it back effectively.
3.  **Simplicity:** It avoids the instability of GANs or the computational cost of Perceptual (VGG) losses, which are overkill for simple geometric shapes.


---

## 📚 References & Novelty

### The Novelty: "Neural ODEs meet V-JEPA"
While based on existing architectures, this implementation introduces a specific combination tailored for **continuous physical dynamics**:

1.  **Multistep VICReg (Trajectory Consistency):**
    *   *Standard approach:* VICReg is typically applied to static pairs (Z_view1, Z_view2).
    *   *Our approach:* We apply VICReg to **unrolled temporal sequences** ($z_t \to z_{t+1} \dots \to z_{t+8}$). This forces the embedding space to be not just "informative" (Covariance) but "dynamically stable" over time.
2.  **Residual Latent Dynamics:**
    *   Instead of predicting states ($z_{t+1}$), we model the *velocity* ($z_{t+1} = z_t + \Delta z$). Effectively, the Predictor learns the differential equation of the latent manifold, similar to a Neural ODE.

### Bibliography

**[1] I-JEPA**
Assran, M., et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture". *CVPR 2023*. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

**[2] V-JEPA**
Bardes, A., et al. (2024). "Revisiting Feature Prediction for Learning Visual Representations from Video". *TMLR 2024*. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)

**[3] VICReg**
Bardes, A., Ponce, J., & LeCun, Y. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning". *ICLR 2022*. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)

**[4] Neural ODEs**
Chen, R. T. Q., et al. (2018). "Neural Ordinary Differential Equations". *NeurIPS 2018*. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

**[5] Cosine Annealing**
Loshchilov, I., & Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts". *ICLR 2017*. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
