import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# 1) Tiny "Data Catalog" + Data Contract metadata
# ----------------------------
CATALOG = [
    {
        "id": "ds_rev_region",
        "title": "Revenue by Region (Gold)",
        "description": "Monthly revenue, region, currency, fx-adjusted, finance certified.",
        "tags": ["revenue", "finance", "region", "monthly", "gold"],
        "required_fields": ["month", "region", "revenue"],
        "completeness": 0.98,
    },
    {
        "id": "ds_rev_raw",
        "title": "Revenue Events (Raw)",
        "description": "Transaction-level revenue events. May have missing region mapping.",
        "tags": ["revenue", "transactions", "raw", "events"],
        "required_fields": ["event_time", "amount"],
        "completeness": 0.72,
    },
    {
        "id": "ds_orders",
        "title": "Orders (Silver)",
        "description": "Order headers with customer and item counts. Not finance certified.",
        "tags": ["orders", "customers", "items", "silver"],
        "required_fields": ["order_id", "order_time"],
        "completeness": 0.90,
    },
    {
        "id": "ds_marketing_spend",
        "title": "Marketing Spend (Gold)",
        "description": "Campaign spend by channel/day with attribution keys.",
        "tags": ["marketing", "spend", "channel", "campaign", "gold"],
        "required_fields": ["date", "channel", "spend"],
        "completeness": 0.95,
    },
    {
        "id": "api_contract_registry",
        "title": "Data Contract Registry API",
        "description": "API to fetch contract schemas, required fields, and validation status.",
        "tags": ["data contract", "schema", "validation", "registry", "api"],
        "required_fields": ["asset_id", "schema", "status"],
        "completeness": 0.99,
    },
    {
        "id": "ds_customer_profile",
        "title": "Customer Profile (PII Restricted)",
        "description": "Customer demographics and identifiers. Access-controlled.",
        "tags": ["customer", "pii", "restricted", "profile"],
        "required_fields": ["customer_id"],
        "completeness": 0.93,
    },
]

def contract_ok(asset, strict: bool) -> bool:
    """
    Toy contract rule:
    - strict: require completeness >= 0.92
    - lenient: allow everything, but we will show warnings
    """
    if not strict:
        return True
    return asset["completeness"] >= 0.92

def contract_warning(asset) -> str:
    if asset["completeness"] >= 0.92:
        return ""
    return f"⚠️ Contract completeness low ({asset['completeness']:.2f}). Missing/unstable fields possible."

# ----------------------------
# 2) "Semantic Search" (toy)
#    We'll use TF-IDF cosine as a stand-in for embeddings.
#    Hybrid = semantic + tag overlap boost + title boost.
# ----------------------------
@st.cache_data
def build_vectorizer(catalog):
    docs = []
    for a in catalog:
        docs.append(f"{a['title']} {a['description']} {' '.join(a['tags'])}")
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(docs)
    return vec, mat

def rank_assets(query, mode: str):
    """
    mode: "vector" or "hybrid"
    returns list of (asset, score)
    """
    vec, mat = build_vectorizer(CATALOG)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).ravel()

    results = []
    q_words = set([w.lower() for w in query.split()])

    for i, a in enumerate(CATALOG):
        score = float(sims[i])

        if mode == "hybrid":
            tag_overlap = len(q_words.intersection(set(a["tags"])))
            title_boost = 0.05 if any(w.lower() in a["title"].lower() for w in q_words) else 0.0
            score = score + 0.03 * tag_overlap + title_boost

        results.append((a, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ----------------------------
# 3) PPO-style policy: choose retrieval "action"
#    Actions represent system decisions.
# ----------------------------
ACTIONS = [
    {"name": "Vector + Lenient + top10", "mode": "vector", "strict": False, "topk": 10},
    {"name": "Hybrid + Lenient + top10", "mode": "hybrid", "strict": False, "topk": 10},
    {"name": "Hybrid + Strict + top10",  "mode": "hybrid", "strict": True,  "topk": 10},
    {"name": "Hybrid + Lenient + top30", "mode": "hybrid", "strict": False, "topk": 30},
    {"name": "Hybrid + Strict + top30",  "mode": "hybrid", "strict": True,  "topk": 30},
]

def featurize_state(query: str, role: str):
    """
    State vector s (small + interpretable):
    - normalized query length
    - contains 'monthly' or 'daily' (time intent)
    - contains 'revenue' or 'spend' (finance intent)
    - role one-hot: analyst / engineer / executive
    """
    q = query.lower()
    qlen = min(len(q.split()), 30) / 30.0
    time_intent = 1.0 if ("monthly" in q or "daily" in q or "week" in q) else 0.0
    finance_intent = 1.0 if ("revenue" in q or "spend" in q or "profit" in q) else 0.0

    roles = ["Analyst", "Engineer", "Executive"]
    role_oh = [1.0 if role == r else 0.0 for r in roles]

    s = np.array([qlen, time_intent, finance_intent] + role_oh, dtype=np.float32)
    return s

class PolicyNet(nn.Module):
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

class ValueNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def init_models():
    in_dim = 3 + 3  # [qlen, time_intent, finance_intent] + role one-hot (3 roles)
    n_actions = len(ACTIONS)
    policy = PolicyNet(in_dim, n_actions)
    value = ValueNet(in_dim)
    opt_policy = optim.Adam(policy.parameters(), lr=2e-3)
    opt_value = optim.Adam(value.parameters(), lr=2e-3)
    return policy, value, opt_policy, opt_value

def choose_action(policy, s_np):
    s = torch.tensor(s_np).unsqueeze(0)  # [1, d]
    with torch.no_grad():
        probs = policy(s).squeeze(0)
    dist = torch.distributions.Categorical(probs)
    a = int(dist.sample().item())
    return a, probs.numpy()

def ppo_update(policy, value, opt_policy, opt_value, s_np, a, reward, old_prob, clip_eps=0.2):
    """
    One-step PPO-style update (toy):
    - advantage A = r - V(s)
    - ratio = pi_new(a|s)/pi_old(a|s)
    - clipped objective for policy
    - value regression to reduce variance
    """
    s = torch.tensor(s_np).unsqueeze(0)  # [1, d]
    a_t = torch.tensor([a])

    # Current probs
    probs = policy(s)
    pi_a = probs[0, a]
    ratio = pi_a / torch.tensor(old_prob)

    # Baseline + advantage
    v = value(s)[0]
    A = torch.tensor(reward) - v.detach()

    # Clipped surrogate
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
    policy_loss = -torch.min(unclipped, clipped)

    # Value loss (baseline)
    value_loss = (value(s)[0] - torch.tensor(reward)) ** 2

    opt_policy.zero_grad()
    policy_loss.backward()
    opt_policy.step()

    opt_value.zero_grad()
    value_loss.backward()
    opt_value.step()

    return float(policy_loss.item()), float(value_loss.item()), float(v.item()), float(ratio.item())

# ----------------------------
# 4) Streamlit UI
# ----------------------------
st.set_page_config(page_title="Semantic Search + Data Contracts + PPO (Toy Demo)", layout="wide")
st.title("Semantic Search + Data Contracts + PPO (Toy Demo)")

st.write(
    """
This is a **toy teaching demo**:
- Semantic search uses **TF-IDF cosine** as a stand-in for embeddings.
- The **policy network** chooses retrieval settings (hybrid vs vector, strict vs lenient contract, top-k).
- You provide feedback (**Good/Bad**) as reward, and the policy updates with **PPO-style clipping** + **value baseline**.
"""
)

if "policy" not in st.session_state:
    st.session_state.policy, st.session_state.value, st.session_state.opt_p, st.session_state.opt_v = init_models()
    st.session_state.last = None

col1, col2 = st.columns([1, 1])

with col1:
    query = st.text_input("Search query", value="monthly revenue by region")
    role = st.selectbox("User role", ["Analyst", "Engineer", "Executive"])
    s_np = featurize_state(query, role)

    st.subheader("State vector s (features)")
    st.code(
        f"s = [query_len_norm={s_np[0]:.2f}, time_intent={s_np[1]:.0f}, finance_intent={s_np[2]:.0f}, "
        f"role_Analyst={s_np[3]:.0f}, role_Engineer={s_np[4]:.0f}, role_Executive={s_np[5]:.0f}]"
    )

    if st.button("Run Search (policy chooses action)"):
        a, probs = choose_action(st.session_state.policy, s_np)
        action = ACTIONS[a]

        # rank + apply contract filter (if strict)
        ranked = rank_assets(query, mode=action["mode"])
        filtered = []
        for asset, score in ranked:
            if contract_ok(asset, strict=action["strict"]):
                filtered.append((asset, score))

        results = filtered[: action["topk"]]

        st.session_state.last = {
            "s_np": s_np,
            "a": a,
            "probs": probs,
            "old_prob": float(probs[a]),
            "action": action,
            "results": results,
        }

with col2:
    st.subheader("Policy πθ(a|s) output probabilities")
    if st.session_state.last is None:
        st.info("Click **Run Search** to see policy probabilities + results.")
    else:
        probs = st.session_state.last["probs"]
        a = st.session_state.last["a"]
        action = st.session_state.last["action"]

        for i, act in enumerate(ACTIONS):
            marker = "✅" if i == a else "  "
            st.write(f"{marker} **{act['name']}** → prob = `{probs[i]:.3f}`")

        st.markdown("---")
        st.subheader("Chosen action a")
        st.write(f"**{action['name']}**")
        st.write(f"- mode: `{action['mode']}`")
        st.write(f"- strict contract: `{action['strict']}`")
        st.write(f"- topk: `{action['topk']}`")

        st.markdown("---")
        st.subheader("Search results")
        results = st.session_state.last["results"]
        if len(results) == 0:
            st.warning("No results after applying strict contract filter.")
        else:
            for asset, score in results[:10]:
                warn = contract_warning(asset)
                st.write(f"**{asset['title']}**  (score={score:.3f})")
                st.caption(asset["description"])
                if warn:
                    st.warning(warn)
                st.write(f"Tags: {', '.join(asset['tags'])}")
                st.write("")

        st.markdown("---")
        st.subheader("Provide reward (simulated feedback)")
        cA, cB = st.columns(2)

        def do_update(reward):
            last = st.session_state.last
            pl, vl, v_pred, ratio = ppo_update(
                st.session_state.policy,
                st.session_state.value,
                st.session_state.opt_p,
                st.session_state.opt_v,
                last["s_np"],
                last["a"],
                reward,
                last["old_prob"],
                clip_eps=0.2,
            )
            st.success("Policy updated with PPO-style clipping.")
            st.write(f"Reward R = **{reward:+.1f}**")
            st.write(f"Value baseline Vϕ(s) prediction = `{v_pred:.3f}`")
            st.write(f"Probability ratio r = π_new(a|s) / π_old(a|s) = `{ratio:.3f}` (clipped around 1±0.2)")
            st.write(f"Policy loss = `{pl:.4f}`, Value loss = `{vl:.4f}`")

        with cA:
            if st.button("👍 Good results (reward = +1)"):
                do_update(+1.0)
        with cB:
            if st.button("👎 Bad results (reward = -1)"):
                do_update(-1.0)

st.markdown("---")
st.subheader("How to read the key formulas")

st.write("**1) πθ(a|s)** → “pi-theta of a given s” = probability of action a in situation s.")
st.write("**2) Vϕ(s)** → “V-phi of s” = baseline prediction of how good state s is (expected reward).")
st.write("**3) r = π_new(a|s) / π_old(a|s)** → “ratio” of new probability to old probability for the chosen action.")
st.write("**4) Clipping** keeps r near 1 (e.g., 1±0.2) so updates don’t change behavior too drastically.")
