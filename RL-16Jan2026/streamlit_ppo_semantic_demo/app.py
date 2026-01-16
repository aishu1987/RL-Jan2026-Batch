import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 1) Tiny "Data Catalog" + Toy Data Contract metadata
# ============================================================
CATALOG = [
    {
        "id": "ds_rev_region",
        "title": "Revenue by Region (Gold)",
        "description": "Monthly revenue, region, currency, fx-adjusted, finance certified.",
        "tags": ["revenue", "finance", "region", "monthly", "gold"],
        "required_fields": ["month", "region", "revenue"],
        "completeness": 0.98,
        "finance_certified": True,
    },
    {
        "id": "ds_rev_raw",
        "title": "Revenue Events (Raw)",
        "description": "Transaction-level revenue events. May have missing region mapping.",
        "tags": ["revenue", "transactions", "raw", "events"],
        "required_fields": ["event_time", "amount"],
        "completeness": 0.72,
        "finance_certified": False,
    },
    {
        "id": "ds_orders",
        "title": "Orders (Silver)",
        "description": "Order headers with customer and item counts. Not finance certified.",
        "tags": ["orders", "customers", "items", "silver"],
        "required_fields": ["order_id", "order_time"],
        "completeness": 0.90,
        "finance_certified": False,
    },
    {
        "id": "ds_marketing_spend",
        "title": "Marketing Spend (Gold)",
        "description": "Campaign spend by channel/day with attribution keys.",
        "tags": ["marketing", "spend", "channel", "campaign", "gold"],
        "required_fields": ["date", "channel", "spend"],
        "completeness": 0.95,
        "finance_certified": True,
    },
    {
        "id": "api_contract_registry",
        "title": "Data Contract Registry API",
        "description": "API to fetch contract schemas, required fields, and validation status.",
        "tags": ["data contract", "schema", "validation", "registry", "api"],
        "required_fields": ["asset_id", "schema", "status"],
        "completeness": 0.99,
        "finance_certified": True,
    },
    {
        "id": "ds_customer_profile",
        "title": "Customer Profile (PII Restricted)",
        "description": "Customer demographics and identifiers. Access-controlled.",
        "tags": ["customer", "pii", "restricted", "profile"],
        "required_fields": ["customer_id"],
        "completeness": 0.93,
        "finance_certified": False,
    },
]

def contract_ok(asset, strict: bool, completeness_threshold: float) -> bool:
    """Toy rule: strict => require completeness >= threshold."""
    if not strict:
        return True
    return asset["completeness"] >= completeness_threshold

def contract_warning(asset, completeness_threshold: float) -> str:
    if asset["completeness"] >= completeness_threshold:
        return ""
    return f"⚠️ Contract completeness low ({asset['completeness']:.2f} < {completeness_threshold:.2f}). Missing/unstable fields possible."

# ============================================================
# 2) Toy Semantic Search (TF-IDF cosine) + Hybrid scoring
# ============================================================
@st.cache_data
def build_vectorizer(catalog):
    docs = [f"{a['title']} {a['description']} {' '.join(a['tags'])}" for a in catalog]
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(docs)
    return vec, mat

def rank_assets(query, mode: str, tag_boost: float, title_boost: float):
    """
    mode: 'vector' or 'hybrid'
    returns list of (asset, score, semantic_score, tag_overlap)
    """
    vec, mat = build_vectorizer(CATALOG)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).ravel()

    q_words = set(w.lower() for w in query.split())
    results = []

    for i, a in enumerate(CATALOG):
        sem = float(sims[i])
        overlap = len(q_words.intersection(set(a["tags"])))
        score = sem

        if mode == "hybrid":
            score = sem + tag_boost * overlap
            if any(w.lower() in a["title"].lower() for w in q_words):
                score += title_boost

        results.append((a, score, sem, overlap))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ============================================================
# 3) PPO-style policy: choose retrieval "action"
# ============================================================
ACTIONS = [
    {"name": "Vector + Lenient + top10", "mode": "vector", "strict": False, "topk": 10},
    {"name": "Hybrid + Lenient + top10", "mode": "hybrid", "strict": False, "topk": 10},
    {"name": "Hybrid + Strict + top10",  "mode": "hybrid", "strict": True,  "topk": 10},
    {"name": "Hybrid + Lenient + top30", "mode": "hybrid", "strict": False, "topk": 30},
    {"name": "Hybrid + Strict + top30",  "mode": "hybrid", "strict": True,  "topk": 30},
]

def featurize_state(query: str, role: str):
    """
    Small + interpretable state vector:
    - normalized query length
    - time intent (monthly/daily/week)
    - finance intent (revenue/spend/profit)
    - role one-hot: Analyst / Engineer / Executive
    """
    q = query.lower()
    qlen = min(len(q.split()), 30) / 30.0
    time_intent = 1.0 if ("monthly" in q or "daily" in q or "week" in q) else 0.0
    finance_intent = 1.0 if ("revenue" in q or "spend" in q or "profit" in q) else 0.0

    roles = ["Analyst", "Engineer", "Executive"]
    role_oh = [1.0 if role == r else 0.0 for r in roles]

    return np.array([qlen, time_intent, finance_intent] + role_oh, dtype=np.float32)

class PolicyNet(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def init_models(lr_policy: float, lr_value: float, hidden: int):
    in_dim = 6
    n_actions = len(ACTIONS)
    policy = PolicyNet(in_dim, n_actions, hidden=hidden)
    value = ValueNet(in_dim, hidden=hidden)
    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_value = optim.Adam(value.parameters(), lr=lr_value)
    return policy, value, opt_policy, opt_value

def choose_action(policy, s_np):
    s = torch.tensor(s_np).unsqueeze(0)
    with torch.no_grad():
        probs = policy(s).squeeze(0)
    dist = torch.distributions.Categorical(probs)
    a = int(dist.sample().item())
    logp = float(dist.log_prob(torch.tensor(a)).item())
    return a, probs.numpy(), logp

def ppo_update(policy, value, opt_policy, opt_value, s_np, a, reward, old_logp, clip_eps=0.2, c_value=0.5, c_entropy=0.01):
    """
    One-state PPO-style update (teaching demo):
      - Advantage A = r - V(s)
      - ratio = exp(logp_new - logp_old)
      - clipped policy objective
      - value regression
      - entropy bonus (encourage exploration)
    """
    s = torch.tensor(s_np).unsqueeze(0)

    # New log prob for the chosen action
    probs = policy(s)
    dist = torch.distributions.Categorical(probs.squeeze(0))
    logp_new = dist.log_prob(torch.tensor(a))
    ratio = torch.exp(logp_new - torch.tensor(old_logp))

    # Baseline + advantage
    v = value(s)[0]
    A = torch.tensor(reward, dtype=torch.float32) - v.detach()

    # PPO clipped surrogate
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
    policy_loss = -torch.min(surr1, surr2)

    # Value loss
    value_loss = (value(s)[0] - torch.tensor(reward, dtype=torch.float32)) ** 2

    # Entropy bonus
    entropy = dist.entropy()
    loss = policy_loss + c_value * value_loss - c_entropy * entropy

    opt_policy.zero_grad()
    opt_value.zero_grad()
    loss.backward()
    opt_policy.step()
    opt_value.step()

    clipped = float((ratio.detach().cpu().item() > (1 + clip_eps)) or (ratio.detach().cpu().item() < (1 - clip_eps)))
    return {
        "loss_total": float(loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value": float(value_loss.item()),
        "entropy": float(entropy.mean().item()),
        "v_pred": float(v.item()),
        "ratio": float(ratio.item()),
        "is_clipped": bool(clipped),
        "logp_new": float(logp_new.item()),
    }

# ============================================================
# 4) Reward shaping (for auto-train demo)
# ============================================================
def auto_reward(query: str, role: str, action_cfg: dict, results: list, completeness_threshold: float):
    """
    A simple synthetic evaluator so students can see iteration logs and curves.

    Reward intuition:
    - Higher semantic score for top results => better
    - If finance intent and top result is finance certified => bonus
    - Strict contract: mild bonus if it picks compliant assets
    - top30: mild penalty (noise)
    """
    q = query.lower()
    finance_intent = ("revenue" in q or "spend" in q or "profit" in q)

    if len(results) == 0:
        return -1.0

    top_asset, top_score, top_sem, top_overlap = results[0]

    r = 0.0
    r += 0.6 * float(top_sem)
    r += 0.1 * float(top_overlap)
    r += 0.1 if action_cfg["mode"] == "hybrid" else 0.0

    if finance_intent and top_asset.get("finance_certified", False):
        r += 0.25

    if action_cfg["strict"]:
        r += 0.10 if top_asset["completeness"] >= completeness_threshold else -0.15

    if action_cfg["topk"] == 30:
        r -= 0.05

    if role == "Executive" and top_asset.get("finance_certified", False):
        r += 0.10
    if role == "Engineer" and ("raw" in top_asset["tags"] or "events" in top_asset["tags"]):
        r += 0.05

    return float(np.tanh(2.0 * r))  # squash to [-1, +1]

# ============================================================
# 5) Streamlit UI
# ============================================================
st.set_page_config(page_title="Semantic Search + Data Contracts + PPO (Interactive)", layout="wide")
st.title("Interactive RL Demo: Semantic Search + Data Contracts + PPO (Toy)")

st.write(
    """
This UI lets you **tweak PPO + retrieval parameters** and see **what action the policy chooses**, the **results**, and a **clear explanation** of the PPO update.
- Semantic search uses **TF‑IDF cosine** (stand‑in for embeddings).
- The policy chooses retrieval settings (vector vs hybrid, strict vs lenient, top‑k).
- You can train by **manual feedback** (Good/Bad) or **auto‑train** with a synthetic evaluator.
"""
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls (tweak in class)")

st.sidebar.subheader("PPO / Model Hyperparameters")
clip_eps = st.sidebar.slider("clip_eps (ε)", 0.05, 0.40, 0.20, 0.01)
lr_policy = st.sidebar.select_slider("policy learning rate", options=[1e-4, 3e-4, 1e-3, 2e-3, 3e-3], value=2e-3)
lr_value  = st.sidebar.select_slider("value learning rate",  options=[1e-4, 3e-4, 1e-3, 2e-3, 3e-3], value=2e-3)
hidden = st.sidebar.selectbox("hidden size", [16, 32, 64, 128], index=1)
c_value = st.sidebar.slider("value loss coef (c1)", 0.1, 1.0, 0.5, 0.05)
c_entropy = st.sidebar.slider("entropy coef (c2)", 0.0, 0.05, 0.01, 0.005)

st.sidebar.subheader("Hybrid Search Weights")
tag_boost = st.sidebar.slider("tag overlap boost", 0.00, 0.10, 0.03, 0.005)
title_boost = st.sidebar.slider("title boost", 0.00, 0.20, 0.05, 0.01)

st.sidebar.subheader("Contract Strictness")
completeness_threshold = st.sidebar.slider("completeness threshold", 0.70, 0.99, 0.92, 0.01)

st.sidebar.subheader("Training Mode")
train_mode = st.sidebar.radio("Feedback mode", ["Manual (Good/Bad)", "Auto-train (synthetic reward)"], index=0)
reward_scale = st.sidebar.slider("Reward scale (for 'return')", 100, 2000, 1000, 50)

if st.sidebar.button("Reset policy / value networks"):
    for k in ["policy", "value", "opt_p", "opt_v", "history", "last"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ---------------- session init ----------------
if "policy" not in st.session_state:
    st.session_state.policy, st.session_state.value, st.session_state.opt_p, st.session_state.opt_v = init_models(
        lr_policy=float(lr_policy), lr_value=float(lr_value), hidden=int(hidden)
    )
    st.session_state.history = []
    st.session_state.last = None

# ---------------- main inputs ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Query + User context")
    query = st.text_input("Search query", value="monthly revenue by region")
    role = st.selectbox("User role", ["Analyst", "Engineer", "Executive"])

    s_np = featurize_state(query, role)
    st.markdown("**State vector $s$ (interpretable features)**")
    st.code(
        f"s = [query_len_norm={s_np[0]:.2f}, time_intent={s_np[1]:.0f}, finance_intent={s_np[2]:.0f}, "
        f"role_Analyst={s_np[3]:.0f}, role_Engineer={s_np[4]:.0f}, role_Executive={s_np[5]:.0f}]"
    )

    st.markdown("**Why does tabular RL not scale here?**")
    st.write("- Continuous or high-dimensional customer features create a massive/infinite state space.")
    st.write("- A neural policy generalizes across similar queries and users via shared parameters θ.")

    st.markdown("---")
    run_once = st.button("Run Search (policy chooses action)")

with col2:
    st.subheader("Policy output + retrieval results")
    if st.session_state.last is None:
        st.info("Click **Run Search** to see policy probabilities, chosen action, and results.")
    else:
        last = st.session_state.last
        probs = last["probs"]
        a = last["a"]
        action = last["action"]

        st.markdown("**Policy probabilities $\\pi_\\theta(a\\mid s)$**")
        for i, act in enumerate(ACTIONS):
            marker = "✅" if i == a else "  "
            st.write(f"{marker} **{act['name']}** → prob = `{probs[i]:.3f}`")

        st.markdown("---")
        st.markdown("**Chosen action**")
        st.write(f"**{action['name']}**")
        st.write(f"- mode: `{action['mode']}`")
        st.write(f"- strict contract: `{action['strict']}`")
        st.write(f"- topk: `{action['topk']}`")

        st.markdown("---")
        st.markdown("**Top results (preview)**")
        results = last["results"]
        if len(results) == 0:
            st.warning("No results after applying strict contract filter.")
        else:
            for asset, score, sem, overlap in results[:10]:
                st.write(f"**{asset['title']}**  (score={score:.3f}, semantic={sem:.3f}, tag_overlap={overlap})")
                st.caption(asset["description"])
                warn = contract_warning(asset, completeness_threshold)
                if warn and action["strict"]:
                    st.warning(warn)
                elif warn:
                    st.info(warn)
                st.write(f"Tags: {', '.join(asset['tags'])}")
                st.write("")

def run_search_and_store():
    a, probs, logp = choose_action(st.session_state.policy, s_np)
    action = ACTIONS[a]

    ranked = rank_assets(query, mode=action["mode"], tag_boost=tag_boost, title_boost=title_boost)

    filtered = []
    for asset, score, sem, overlap in ranked:
        if contract_ok(asset, strict=action["strict"], completeness_threshold=completeness_threshold):
            filtered.append((asset, score, sem, overlap))

    results = filtered[: action["topk"]]
    st.session_state.last = {
        "s_np": s_np,
        "a": a,
        "probs": probs,
        "old_logp": logp,
        "action": action,
        "results": results,
    }

if run_once:
    run_search_and_store()
    st.rerun()

# ---------------- training / feedback panel ----------------
st.markdown("---")
st.subheader("Train the policy (feedback → PPO update)")

if st.session_state.last is None:
    st.info("Run a search first, then provide feedback to update the policy.")
else:
    last = st.session_state.last

    with st.expander("Explain the training output (Iter / recent mean return / blocks)", expanded=True):
        st.write(
            """
**What is an Iteration (Iter)?**  
One **Iter** is one update step. In full PPO: rollout → advantages → SGD updates.
In this demo: **one feedback update** = one Iter.

**What is “return”?**  
We display `return = reward × reward_scale` so numbers look like typical RL logs (hundreds/thousands).

**What is “recent mean return (last 10 blocks)”?**  
A moving average of the last 10 returns, used because RL rewards are noisy.
"""
        )

    if train_mode.startswith("Manual"):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            good = st.button("👍 Good (reward = +1)")
        with c2:
            bad = st.button("👎 Bad (reward = -1)")
        with c3:
            custom = st.slider("Or give a custom reward (continuous)", -1.0, 1.0, 0.0, 0.1)

        reward = None
        if good:
            reward = 1.0
        elif bad:
            reward = -1.0
        elif abs(custom) > 1e-9:
            reward = float(custom)

        if reward is not None:
            upd = ppo_update(
                st.session_state.policy,
                st.session_state.value,
                st.session_state.opt_p,
                st.session_state.opt_v,
                last["s_np"],
                last["a"],
                reward,
                last["old_logp"],
                clip_eps=float(clip_eps),
                c_value=float(c_value),
                c_entropy=float(c_entropy),
            )
            ret = reward * reward_scale

            st.session_state.history.append({
                "iter": len(st.session_state.history) + 1,
                "reward": reward,
                "return": ret,
                **upd,
            })

            st.success("Updated policy with PPO-style clipping + value baseline.")
            st.markdown("**Update explanation (what changed?)**")
            st.write(f"- Reward R = **{reward:+.2f}** (your feedback)")
            st.write(f"- Value baseline Vϕ(s) = `{upd['v_pred']:.3f}` → advantage roughly `A = R - V`")
            st.write(f"- Ratio r = exp(logπ_new − logπ_old) = `{upd['ratio']:.3f}`")
            st.write(f"- Clipped? `{upd['is_clipped']}` (True = PPO prevented too-large update)")
            st.write(f"- Policy loss = `{upd['loss_policy']:.4f}`, Value loss = `{upd['loss_value']:.4f}`, Entropy = `{upd['entropy']:.4f}`")

    else:
        st.write("Auto-train uses a synthetic evaluator so students can see iteration logs and curves.")
        n = st.slider("How many iterations to run", 5, 200, 40, 5)
        run_auto = st.button("Run auto-train now")

        if run_auto:
            for _ in range(n):
                run_search_and_store()
                last = st.session_state.last
                reward = auto_reward(query, role, last["action"], last["results"], completeness_threshold)
                upd = ppo_update(
                    st.session_state.policy,
                    st.session_state.value,
                    st.session_state.opt_p,
                    st.session_state.opt_v,
                    last["s_np"],
                    last["a"],
                    reward,
                    last["old_logp"],
                    clip_eps=float(clip_eps),
                    c_value=float(c_value),
                    c_entropy=float(c_entropy),
                )
                ret = reward * reward_scale
                st.session_state.history.append({
                    "iter": len(st.session_state.history) + 1,
                    "reward": float(reward),
                    "return": float(ret),
                    **upd,
                })

            st.success(f"Auto-trained for {n} iterations.")
            st.rerun()

# ---------------- plots + logs ----------------
st.markdown("---")
st.subheader("Learning curves + console-style logs")

hist = st.session_state.history
if len(hist) == 0:
    st.info("No training history yet. Provide feedback or run auto-train.")
else:
    returns = [h["return"] for h in hist]
    entropies = [h["entropy"] for h in hist]
    ratios = [h["ratio"] for h in hist]

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Total updates", value=len(hist))
    with cB:
        st.metric("Latest return", value=f"{returns[-1]:.2f}")
    with cC:
        last10 = returns[-10:]
        st.metric("Recent mean return (last 10)", value=f"{np.mean(last10):.2f}")

    st.line_chart({"return": returns})
    st.line_chart({"entropy": entropies})
    st.line_chart({"ratio": ratios})

    st.markdown("**Console-style summary (every 10 iterations):**")
    lines = []
    for i in range(10, len(returns) + 1, 10):
        window = returns[max(0, i - 10): i]
        lines.append(f"Iter {i:>3d} | recent mean return (last 10 blocks) = {np.mean(window):>9.2f}")
    st.code("\n".join(lines))

# ---------------- formula reading ----------------
st.markdown("---")
st.subheader("How to read the key formulas (for students)")

st.write("**1) $\\pi_\\theta(a\\mid s)$** → “pi-theta of a given s” = probability of choosing action a in situation s.")
st.write("**2) $V_\\phi(s)$** → “V-phi of s” = baseline prediction of how good state s is (expected reward).")
st.write("**3) $r=\\exp(\\log\\pi_{new}-\\log\\pi_{old})$** → ratio of new to old probability for the chosen action.")
st.write("**4) PPO clipping** keeps r near 1 (e.g., 1±ε) so updates don’t change behavior too drastically.")
