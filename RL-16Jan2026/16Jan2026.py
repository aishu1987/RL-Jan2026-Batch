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