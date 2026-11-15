# Mathematical Foundations & Invariant Proofs in VeritasRL

This document states and proves the exact mathematical invariants enforced in `veritas-rl` across code, C++20 kernels (`csrc/veritas_kernels.cpp`), and randomized `Hypothesis` property tests (`tests/`).

---

## 1. Critic-Free GRPO & DAPO Asymmetric Policy Optimization

Standard Proximal Policy Optimization (PPO) maintains a separate critic network $V_\psi(s_t)$ of equal parameter count to the policy $\pi_\theta$, doubling activation and optimizer-state memory. **Group Relative Policy Optimization (GRPO; Shao et al., DeepSeekMath / DeepSeek-R1)** eliminates the critic by sampling a group of $G$ trajectories $\{o_1, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$ for each prompt $q$ and scoring each trajectory with a deterministic symbolic verifier $r_i = R_{\text{sym}}(q, o_i) \in [0, 1]$:

$$\hat{A}_i = \frac{r_i - \mu_G}{\sqrt{\sigma_G^2 + \varepsilon}}, \qquad \mu_G = \frac{1}{G}\sum_{j=1}^{G} r_j, \qquad \sigma_G^2 = \frac{1}{G}\sum_{j=1}^{G} (r_j - \mu_G)^2$$

**Proposition 1 (Zero-Mean Group Baseline).** For every prompt group of size $G \ge 2$ and $\varepsilon > 0$, $\sum_{i=1}^G \hat{A}_i = 0$. Consequently, constant prompt-difficulty offsets cancel analytically without a learned value baseline.

Following **DAPO (Yu et al., 2025)**, `veritas-rl` decouples the lower and upper clipping thresholds ($\varepsilon_{\text{low}} = 0.20 < \varepsilon_{\text{high}} = 0.28$) on the token likelihood ratio $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ to prevent entropy collapse on low-probability exploratory reasoning tokens:

$$\mathcal{J}_{\text{GRPO-DAPO}}(\theta) = \mathbb{E}_{q, \{o_i\}}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left(\min\!\Big(\rho_{i,t}\hat{A}_i,\; \operatorname{clip}(\rho_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\hat{A}_i\Big) - \beta\, \hat{D}_{\text{KL}}^{(k_3)}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$

---

## 2. Exact Non-Negativity & Unbiasedness of Schulman's $k_3$ KL Estimator

Naive Monte-Carlo KL estimation $k_1 = \log \pi_\theta(a \mid s) - \log \pi_{\text{ref}}(a \mid s)$ takes negative values on individual tokens and exhibits high variance. `veritas-rl` implements **Schulman's $k_3$ estimator** in both C++20 and PyTorch:

$$k_3(a \mid s) = \exp\!\left(\log \frac{\pi_{\text{ref}}(a \mid s)}{\pi_\theta(a \mid s)}\right) - \log \frac{\pi_{\text{ref}}(a \mid s)}{\pi_\theta(a \mid s)} - 1$$

**Theorem 1 (Pointwise Non-Negativity & Unbiasedness).**
1. *Pointwise Non-Negativity:* Let $u = \log \frac{\pi_{\text{ref}}(a \mid s)}{\pi_\theta(a \mid s)} \in \mathbb{R}$. By strict convexity of $e^u$, its tangent line at $u=0$ satisfies $e^u \ge 1 + u$ for all $u \in \mathbb{R}$, with equality iff $u = 0$ ($\pi_\theta(a \mid s) = \pi_{\text{ref}}(a \mid s)$). Thus $k_3(a \mid s) = e^u - u - 1 \ge 0$ everywhere.
2. *Exact KL Expectation:* Taking expectation over $a \sim \pi_\theta(\cdot \mid s)$:
$$\mathbb{E}_{a \sim \pi_\theta}[k_3(a \mid s)] = \sum_a \pi_\theta(a \mid s)\!\left(\frac{\pi_{\text{ref}}(a \mid s)}{\pi_\theta(a \mid s)} - \log\frac{\pi_{\text{ref}}(a \mid s)}{\pi_\theta(a \mid s)} - 1\right) = 1 + D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) - 1 = D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

---

## 3. DoRA Weight Decomposition & Eckart-Young Rank Separation

Given frozen pretrained weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, standard rank-$r$ LoRA parameterizes $W_{\text{LoRA}} = W_0 + \frac{\alpha}{r} B A$. Suppose downstream post-training rescales output neuron magnitudes via $W^\star = \operatorname{diag}(c)\, W_0$ with distinct entries $c \neq \mathbf{1}$. Since $( \operatorname{diag}(c) - I ) W_0$ generically has **full rank** $\min(d_{\text{out}}, d_{\text{in}})$, the **Eckart-Young-Mirsky theorem** lower-bounds LoRA's Frobenius error by $\sum_{k=r+1}^{d} \sigma_k^2((\operatorname{diag}(c)-I)W_0) > 0$.

**DoRA (Liu et al., ICML 2024 Oral)** decouples magnitude $m \in \mathbb{R}^{d_{\text{out}}}$ from directional low-rank updates:

$$W_{\text{DoRA}} = \operatorname{diag}(m)\, \frac{W_0 + \frac{\alpha}{r} B A}{\|W_0 + \frac{\alpha}{r} B A\|_{\text{row}, 2}}$$

1. **Exact Initialization:** At $B = 0$ and $m_i = \|W_{0,i}\|_2$, $W_{\text{DoRA}} = W_0$ identically.
2. **Full-Rank Magnitude Expressivity:** Setting $B = 0$ and $m_i = c_i \|W_{0,i}\|_2$ recovers $W^\star = \operatorname{diag}(c) W_0$ with **zero error** using only $d_{\text{out}}$ parameters.
3. **Zero-Latency Inference Folding:** At serving time, $W_{\text{merged}} \leftarrow W_{\text{DoRA}}$ is materialized once into a standard `nn.Linear` weight tensor (`max |Δ| = 0.00000000` in float32).

---

## 4. Associative Online-Softmax Ring-Attention & Megatron 2D TP

### 4.1 Online-Softmax Ring-Attention (`parallel_ops.py` & `csrc/veritas_kernels.cpp`)
Partition sequence length $S$ across $P$ ring ranks ($S_{\text{local}} = S / P$). As $(K_p, V_p)$ blocks circulate around the ring, each rank maintains running state $(O^{(p)}, m^{(p)}, \ell^{(p)})$ updated via:

$$m_{\text{new}} = \max(m_{\text{old}}, \max_j S_{:,j}), \qquad \ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}}\ell_{\text{old}} + \sum_j e^{S_{:,j} - m_{\text{new}}}$$
$$O_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} O_{\text{old}} + e^{S - m_{\text{new}}} V_p$$

Telescoping $e^{m^{(p-1)} - m^{(P)}}$ across $p = 1, \dots, P$ cancels all intermediate maxima $m^{(p)}$, leaving $O^{(P)} / \ell^{(P)} = \operatorname{Softmax}(Q_r K^\top / \sqrt{d} + M_{\text{causal}}) V$ with $O(S^2 / P)$ per-rank memory instead of $O(S^2)$.

### 4.2 Megatron-LM Column + Row 2D Tensor Parallelism (`MegatronMLP2D`)
Partition $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ row-wise ($W_1 = [W_1^{(1)}; \dots; W_1^{(P)}]$, *Column-Parallel* in $X W_1^\top$) and $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ column-wise ($W_2 = [W_2^{(1)}, \dots, W_2^{(P)}]$, *Row-Parallel* in $H W_2^\top$). Because elementwise $\operatorname{GeLU}$ commutes with block concatenation:

$$\sum_{p=1}^P \operatorname{GeLU}\!\left(X W_1^{(p)\top}\right) W_2^{(p)\top} = \operatorname{GeLU}(X W_1^\top) W_2^\top$$

requiring **zero communication** before the nonlinearity and exactly **one `All-Reduce(SUM)`** in the forward pass (and its dual identity/all-reduce in `autograd.backward`).
