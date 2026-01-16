# Class Notes — 16 Jan 2026  
## Scaling RL with Neural Networks & PPO (Proximal Policy Optimization)

### What this class covers
When the **state space** or **action space** is too large for a Q-table, we use **neural networks** as function approximators.

You’ll learn:
- Why tabular RL breaks at scale (and what “scale” means in practice)
- How **policy networks** (actor) and **value networks** (critic) replace tables
- PPO: the **clipped objective**, **value baseline**, and why it stabilizes training
- How to **visualize reward curves** and interpret learning signals
- A complete **Python (PyTorch) PPO example** in a realistic “Retail Coupon” setting

---

## 1) Why “Scaling” is hard in RL
### 1.1 Tabular RL does not scale
In tabular Q-learning, we store a number for every (state, action):

\[
Q(s,a) \in \mathbb{R}
\]

If you have:
- 1,000,000 possible states
- 20 actions

Then your table needs **20 million** Q-values. If state features are continuous (like “time since last purchase”), the number of distinct states becomes effectively infinite.

### 1.2 Neural networks as function approximators
Instead of storing a table, we learn a function:

\[
Q_\theta(s,a) \approx Q(s,a)
\]

or directly learn a **policy**:

\[
\pi_\theta(a \mid s)
\]

Here, \(\theta\) are the neural network parameters (weights).

---

## 2) Quick bridge: from “small MDP” to “feature-based state”
In our earlier toy Retail Coupon MDP, we had a small set of states like:
- LOYAL
- PRICE_SENSITIVE
- COUPON_ADDICT

To scale this up, we represent customer state using **features**:

Example state vector \(s\):
- **R**: recency (days since last purchase)
- **F**: purchase frequency (last 30 days)
- **M**: monetary value (avg basket size)
- **S**: coupon sensitivity score (0 to 1)
- **A**: “addiction” tendency score (0 to 1)

Now \(s\) is a numeric vector:  
\[
s \in \mathbb{R}^d
\]

This is the key change that makes neural networks natural.

---

## 3) Actor–Critic idea (policy + value)
### 3.1 The actor (policy network)
The actor outputs a probability distribution over actions.

If actions are discrete (e.g., coupon level = {0%, 5%, 10%, 20%}), then:

\[
\pi_\theta(a \mid s) = \text{softmax}(f_\theta(s))_a
\]

**How to read it (plain English):**  
“pi-theta of a given s” is the probability that the policy (with parameters theta) chooses action a when it sees state s.

### 3.2 The critic (value network)
The critic predicts the expected future return from a state:

\[
V_\phi(s) \approx \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_{t} \,\middle|\, s_0=s\right]
\]

**Value baseline:** Using \(V(s)\) reduces variance in policy gradients because we measure “how much better than expected” an action was.

---

## 4) Advantages and why they matter
We want a quantity that tells us whether an action was better than the baseline:

\[
A(s_t,a_t) = Q(s_t,a_t) - V(s_t)
\]

In practice, we estimate advantage from rollouts using **GAE(λ)** (Generalized Advantage Estimation):

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

\[
A_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
\]

**How to read GAE quickly:**  
- First compute the one-step “surprise” \(\delta_t\) (TD error)  
- Then add discounted future TD errors with factor \((\gamma\lambda)\)

---

## 5) PPO: the core idea
PPO is a **policy gradient** method designed to make updates **safe** (not too large).

### 5.1 Policy gradient reminder (why ratios appear)
We update the policy to increase the probability of actions that had positive advantage.

PPO compares:
- old policy \(\pi_{\theta_{\text{old}}}\)
- new policy \(\pi_\theta\)

Using the probability ratio:

\[
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
\]

**How to read it:**  
“r-t of theta equals probability under new policy divided by probability under old policy for the action we actually took.”

- If \(r_t > 1\), the new policy increases probability of that action.
- If \(r_t < 1\), the new policy decreases probability.

### 5.2 The PPO clipped objective (the famous formula)
PPO maximizes:

\[
L^{\text{CLIP}}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]
\]

**How to read this safely (step-by-step):**
1. Compute \(r_t(\theta)A_t\): “improvement estimate”
2. Clip the ratio \(r_t\) into \([1-\epsilon, 1+\epsilon]\)
3. Use the **minimum** of clipped and unclipped objective  
   → prevents the optimizer from taking an update that changes action probabilities too much

**Intuition:**  
PPO is like saying: “Improve policy, but don’t move too far in one step.”

### 5.3 Value loss and entropy bonus
In practice, PPO trains actor and critic together with a combined loss:

- Policy loss: \(-L^{\text{CLIP}}\)
- Value loss (critic regression):
\[
L^{V}(\phi) = \mathbb{E}\left[(V_\phi(s_t) - \hat{R}_t)^2\right]
\]
- Entropy bonus (encourage exploration):
\[
H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s)\log \pi_\theta(a|s)
\]

Overall loss:
\[
\mathcal{L} = -L^{\text{CLIP}} + c_1 L^V - c_2 H
\]

Where:
- \(c_1\) controls critic strength
- \(c_2\) controls exploration strength

---

## 6) PPO training loop (what happens “inside an episode”)
A typical PPO cycle:

1. **Rollout** policy for \(T\) steps (collect trajectories)
2. Compute returns \(\hat{R}_t\) and advantages \(A_t\)
3. Do **K epochs** of minibatch SGD on the PPO objective
4. Replace “old policy” with the updated one
5. Repeat

Why PPO updates inside episodes (or across short rollouts):
- We need fresh samples from the current policy (on-policy)
- Large steps can break training → clipping helps, but we still update gradually

---

## 7) Reading reward curves (and what to watch for)
You will plot:
- **Average episodic return**
- **Moving average** (smooth trend)
- (Optional) actor loss / critic loss / entropy

Interpretation guide:
- Reward curve rising steadily → learning is stable
- Reward jumps + collapses → learning rate too high / poor advantage estimates
- Reward flat → weak signal, too much noise, or environment too hard
- Entropy dropping too fast → policy becomes deterministic too early

---

# 8) Hands-on Python: PPO for Retail Coupon (scaled state)

> This is a **self-contained teaching example**.  
> It’s not meant to beat benchmarks—it's meant to make PPO mechanics crystal clear.

## 8.1 Install
```bash
pip install torch numpy matplotlib
```

## 8.2 Full code (single file)
Save as `ppo_retail_scaled.py` and run it.

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------------------------------
# Environment: Scaled Retail Coupon Simulation
# -------------------------------------------------
# State = [recency_days, freq_30d, avg_basket, coupon_sens, addiction]
# Actions = coupon level: 0%, 5%, 10%, 20%
ACTIONS = np.array([0.00, 0.05, 0.10, 0.20], dtype=np.float32)

class RetailCouponEnv:
    """
    A simple stochastic environment that mimics coupon effects:
      - Coupon can increase purchase probability in the short-term
      - Frequent couponing can increase 'addiction' (future dependence)
      - Profit = margin - coupon_cost if purchased; 0 if no purchase
    """
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        # Initialize customer features
        recency = self.rng.integers(0, 30)           # days since last purchase
        freq = self.rng.integers(0, 15)              # purchases in last 30 days
        basket = self.rng.uniform(10.0, 120.0)       # average basket size
        sens = self.rng.uniform(0.0, 1.0)            # coupon sensitivity
        addiction = self.rng.uniform(0.0, 0.3)       # starts low
        self.s = np.array([recency, freq, basket, sens, addiction], dtype=np.float32)
        return self._obs()

    def _obs(self):
        # Normalize for NN stability
        recency, freq, basket, sens, addiction = self.s
        return np.array([
            recency / 30.0,
            freq / 15.0,
            basket / 120.0,
            sens,
            addiction
        ], dtype=np.float32)

    def step(self, a_idx):
        coupon = float(ACTIONS[a_idx])
        recency, freq, basket, sens, addiction = self.s

        # Base purchase probability decreases with recency, increases with freq & sens
        # and is influenced by addiction and coupon level.
        # Use a logistic model to keep it between (0,1).
        x = (
            -1.2 * (recency / 30.0) +
             1.0 * (freq / 15.0) +
             1.2 * sens +
             0.8 * addiction +
             3.5 * coupon * (0.4 + 0.6 * sens)   # coupon lift higher for sensitive users
        )
        p_buy = 1.0 / (1.0 + math.exp(-x))
        buy = (self.rng.random() < p_buy)

        # Economics
        margin = 0.25 * basket  # margin is 25% of basket
        coupon_cost = coupon * basket  # discount cost if purchase happens
        reward = (margin - coupon_cost) if buy else 0.0

        # Dynamics: update state
        if buy:
            recency = 0
            freq = min(freq + 1, 15)
            # basket fluctuates
            basket = float(np.clip(basket * self.rng.normal(1.0, 0.05), 10.0, 120.0))
        else:
            recency = min(recency + 1, 30)
            # freq decays slowly if no purchase
            freq = max(freq - (1 if self.rng.random() < 0.15 else 0), 0)
            basket = float(np.clip(basket * self.rng.normal(0.995, 0.03), 10.0, 120.0))

        # Addiction: grows if you give coupons often (especially high coupon)
        # but can decay slowly if you stop.
        addiction = float(np.clip(addiction + 0.10 * coupon - (0.01 if coupon == 0.0 else 0.0), 0.0, 1.0))

        # Sensitivity drifts slowly
        sens = float(np.clip(sens + self.rng.normal(0.0, 0.01), 0.0, 1.0))

        self.s = np.array([recency, freq, basket, sens, addiction], dtype=np.float32)

        done = False  # continuing task; we'll cut rollouts ourselves
        info = {"p_buy": p_buy, "buy": buy, "coupon": coupon}
        return self._obs(), float(reward), done, info

# -------------------------------------------------
# Actor-Critic Network
# -------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

# -------------------------------------------------
# PPO utilities
# -------------------------------------------------
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T+1]  (bootstrap value at T)
    returns: advantages [T], returns [T]
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret

# -------------------------------------------------
# Main training
# -------------------------------------------------
def train_ppo(
    steps_per_iter=2048,
    iters=80,
    epochs=10,
    minibatch=256,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.01,
    seed=0
):
    env = RetailCouponEnv(seed=seed)
    obs_dim = 5
    act_dim = len(ACTIONS)

    device = torch.device("cpu")
    net = ActorCritic(obs_dim, act_dim).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    ep_returns = []
    moving = []

    obs = env.reset()
    running_return = 0.0

    for it in range(1, iters + 1):
        # Buffers
        obs_buf = np.zeros((steps_per_iter, obs_dim), dtype=np.float32)
        act_buf = np.zeros(steps_per_iter, dtype=np.int64)
        logp_buf = np.zeros(steps_per_iter, dtype=np.float32)
        rew_buf = np.zeros(steps_per_iter, dtype=np.float32)
        val_buf = np.zeros(steps_per_iter + 1, dtype=np.float32)

        # Rollout
        for t in range(steps_per_iter):
            obs_buf[t] = obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, value = net(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            act = int(action.item())
            next_obs, reward, done, info = env.step(act)

            act_buf[t] = act
            logp_buf[t] = float(logp.item())
            rew_buf[t] = float(reward)
            val_buf[t] = float(value.item())

            running_return += reward
            obs = next_obs

            # We create artificial "episodes" for reporting
            if (t + 1) % 256 == 0:
                ep_returns.append(running_return)
                running_return = 0.0

        # Bootstrap value for GAE
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, v_last = net(obs_t)
        val_buf[-1] = float(v_last.item())

        # Compute advantage + returns
        adv, ret = compute_gae(rew_buf, val_buf, gamma=gamma, lam=lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Torch tensors
        obs_tensor = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_tensor = torch.tensor(act_buf, dtype=torch.int64, device=device)
        logp_old = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_tensor = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(ret, dtype=torch.float32, device=device)

        # PPO updates
        idxs = np.arange(steps_per_iter)
        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, steps_per_iter, minibatch):
                mb = idxs[start:start+minibatch]

                logits, value = net(obs_tensor[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_tensor[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_old[mb])
                surr1 = ratio * adv_tensor[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_tensor[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((value - ret_tensor[mb]) ** 2).mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        # Logging
        if len(ep_returns) >= 10:
            m = float(np.mean(ep_returns[-10:]))
        else:
            m = float(np.mean(ep_returns))
        moving.append(m)

        if it % 10 == 0:
            print(f"Iter {it:3d} | recent mean return (last 10 blocks) = {m:8.2f}")

    return ep_returns, moving

if __name__ == "__main__":
    ep_returns, moving = train_ppo()

    # Plot reward curve
    plt.figure()
    plt.plot(ep_returns, label="Return per 256-step block")
    plt.plot(np.linspace(0, len(ep_returns)-1, len(moving)), moving, label="Moving mean (last 10)")
    plt.xlabel("Block")
    plt.ylabel("Return")
    plt.title("PPO Learning Curve — Retail Coupon (Scaled State)")
    plt.legend()
    plt.show()
```

---

## 9) What to tweak in class (fast experiments)
Try changing one knob at a time:
- Increase clip \(\epsilon\) from 0.2 → 0.3 (riskier updates)
- Reduce entropy coefficient \(c_2\) (less exploration)
- Increase steps_per_iter (better advantage estimates)
- Reduce learning rate (more stable, slower learning)

---

## 10) Common failure patterns (and how to debug)
- **Policy collapses too early** (entropy goes to 0 fast)  
  → increase entropy bonus, reduce LR
- **Reward oscillates wildly**  
  → reduce LR, increase batch size, reduce clip eps
- **Value loss explodes**  
  → normalize rewards, check return computation
- **No learning**  
  → advantage is near-zero; environment signal too weak; fix reward shaping

---

## 11) Summary you should remember
- Neural networks let RL handle huge/continuous state spaces.
- PPO is stable because it limits how much the policy changes per update (clipping).
- Value baseline + GAE reduce variance and improve learning.
- Reward curves are your first diagnostic tool—plot them early and often.

---

### Infographic (included with this class package)
See: `Class_16Jan2026_PPO_Infographic.png`
