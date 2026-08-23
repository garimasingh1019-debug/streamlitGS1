"""
Concept Explainer — a Streamlit app that lets you bring your own OpenAI or
Anthropic (Claude) API key, browse available models with pricing, and get
multi-audience explanations of any technical concept (with analogies,
business impact, and follow-up Q&A).

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push this repo (app.py + requirements.txt) to GitHub and point
    Streamlit Cloud at it. No secrets are required — each user pastes
    their own API key into the sidebar at runtime; it is kept only in
    that browser session's memory (st.session_state) and is never
    written to disk or logged.
"""

import io
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Optional SDK imports — the app should still load (and tell the user what's
# missing) even if one provider's SDK isn't installed yet.
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Static pricing reference (USD per 1M tokens). Neither provider's API
# returns live pricing, so this table is maintained by hand. Prices change
# often — treat these as *reference* numbers and verify against the
# provider's official pricing page before making cost decisions.
#   Anthropic: https://www.anthropic.com/pricing#api
#   OpenAI:    https://platform.openai.com/docs/pricing
# Last checked: August 2026.
# Matching is done by exact model id first, then by longest matching prefix,
# so new dated snapshots of a known model (e.g. "-20260101" suffixes) still
# resolve to a sensible price.
# ---------------------------------------------------------------------------
PRICING_USD_PER_1M = {
    "anthropic": {
        "claude-opus-4-8": (5.00, 25.00),
        "claude-opus-4-7": (5.00, 25.00),
        "claude-opus-4-6": (5.00, 25.00),
        "claude-sonnet-5": (2.00, 10.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-fable-5": (10.00, 50.00),
        "claude-mythos-5": (10.00, 50.00),
        "claude-3-5-haiku": (0.80, 4.00),
    },
    "openai": {
        "gpt-5.6-sol": (5.00, 30.00),
        "gpt-5.6-terra": (2.00, 12.00),
        "gpt-5.6-luna": (0.20, 1.20),
        "gpt-5.5": (5.00, 30.00),
        "gpt-5.4": (1.25, 7.50),
        "gpt-5-nano": (0.05, 0.40),
        "gpt-5-mini": (0.25, 2.00),
        "gpt-5": (1.25, 10.00),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    },
}


def lookup_price(provider: str, model_id: str):
    """Return (input_$/1M, output_$/1M) for a model id, or None if unknown.
    Falls back to the longest known prefix match so dated / regional model
    id variants still resolve."""
    table = PRICING_USD_PER_1M.get(provider, {})
    if model_id in table:
        return table[model_id]
    best_match, best_len = None, 0
    for key, price in table.items():
        if model_id.startswith(key) and len(key) > best_len:
            best_match, best_len = price, len(key)
    return best_match


# ---------------------------------------------------------------------------
# System prompt that drives the multi-audience explanation style
# ---------------------------------------------------------------------------
EXPLAINER_SYSTEM_PROMPT = """You are a world-class technical educator and \
business communicator. Your job is to explain technology concepts to a \
mixed audience in one response: non-technical stakeholders / SMEs (subject \
matter experts from a non-tech domain), domain experts, and technical / \
engineering practitioners.

When the user gives you one concept or a set of concepts/keywords, ALWAYS \
structure your answer in markdown with exactly these sections, in this \
order:

## In Plain English
A short, jargon-free explanation anyone could understand, built around one \
or two vivid, accurate analogies from everyday life. No dumbing-down of \
facts — just plain language.

## For Domain Experts / SMEs
Connect the concept to the vocabulary, workflows, and concerns of a \
subject-matter expert who isn't a software engineer (e.g. a business \
analyst, clinician, operations lead, or industry specialist). Show how it \
maps onto things they already know.

## For Technical Practitioners
A precise, technically rigorous explanation: mechanisms, architecture, key \
terminology, relevant algorithms/standards, common pitfalls, and how it \
compares to related approaches. Do not oversimplify here — assume a \
competent engineer as the reader.

## Business Impact
Why this matters commercially: cost/efficiency effects, competitive \
advantage, risk, ROI drivers, and realistic use cases across industries.

## Key Takeaways
3-6 crisp bullet points summarizing the above.

Rules:
- Keep every section accurate and internally consistent — the plain-English \
version must not contradict the technical version, just express it more \
simply.
- Prefer concrete, checkable claims over vague hype.
- If the user asks a specific follow-up question afterward, answer it \
directly and concisely; you do not need to repeat the full section \
structure for follow-ups, only for the first explanation of a new \
concept.
"""


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "provider": "OpenAI",
        "api_key": "",
        "validated": False,
        "models": [],           # list of dicts: {"id": ..., "input": x, "output": y}
        "selected_model": None,
        "messages": [],         # [{"role": "user"/"assistant", "content": str}]
        "validation_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
st.set_page_config(page_title="Concept Explainer", page_icon="💡", layout="wide")


# ---------------------------------------------------------------------------
# Provider helpers: validate key + list models
# ---------------------------------------------------------------------------
def validate_and_list_openai(api_key: str):
    if not OPENAI_SDK_AVAILABLE:
        raise RuntimeError("The `openai` package isn't installed. Add it to requirements.txt.")
    client = OpenAI(api_key=api_key)
    # A lightweight call that both authenticates the key and returns models.
    resp = client.models.list()
    ids = sorted(m.id for m in resp.data)
    # Keep this list from becoming enormous / noisy: prioritize chat-capable
    # families a user is likely to want for text generation.
    preferred_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt")
    filtered = [m for m in ids if m.startswith(preferred_prefixes)]
    return filtered or ids


def validate_and_list_anthropic(api_key: str):
    if not ANTHROPIC_SDK_AVAILABLE:
        raise RuntimeError("The `anthropic` package isn't installed. Add it to requirements.txt.")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.models.list(limit=100)
    ids = sorted((m.id for m in resp.data), reverse=True)
    return ids


def build_model_rows(provider_key: str, model_ids):
    rows = []
    for mid in model_ids:
        price = lookup_price(provider_key, mid)
        if price:
            rows.append({"id": mid, "input": price[0], "output": price[1]})
        else:
            rows.append({"id": mid, "input": None, "output": None})
    return rows


# ---------------------------------------------------------------------------
# Chat call helpers
# ---------------------------------------------------------------------------
def call_openai(api_key, model, system_prompt, history):
    client = OpenAI(api_key=api_key)
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


def call_anthropic(api_key, model, system_prompt, history):
    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    resp = client.messages.create(
        model=model,
        system=system_prompt,
        max_tokens=2000,
        messages=messages,
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def call_model(history):
    provider = st.session_state.provider
    model = st.session_state.selected_model
    key = st.session_state.api_key
    if provider == "OpenAI":
        return call_openai(key, model, EXPLAINER_SYSTEM_PROMPT, history)
    else:
        return call_anthropic(key, model, EXPLAINER_SYSTEM_PROMPT, history)


# ---------------------------------------------------------------------------
# Sidebar — provider, key, validation, model + pricing table
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    provider = st.radio(
        "Provider",
        options=["OpenAI", "Anthropic (Claude)"],
        index=0 if st.session_state.provider == "OpenAI" else 1,
        horizontal=True,
    )
    provider_key = "openai" if provider == "OpenAI" else "anthropic"
    normalized_provider = "OpenAI" if provider == "OpenAI" else "Anthropic"

    if normalized_provider != st.session_state.provider:
        # Provider switched — reset validation/model state, keep chat.
        st.session_state.provider = normalized_provider
        st.session_state.validated = False
        st.session_state.models = []
        st.session_state.selected_model = None

    api_key_input = st.text_input(
        f"{provider} API key",
        type="password",
        value=st.session_state.api_key,
        help="Stored only in this browser session's memory — never saved to disk.",
    )

    validate_clicked = st.button("🔑 Validate & fetch models", use_container_width=True)

    if validate_clicked:
        st.session_state.api_key = api_key_input
        st.session_state.validation_error = ""
        if not api_key_input.strip():
            st.session_state.validation_error = "Please enter an API key first."
            st.session_state.validated = False
        else:
            try:
                with st.spinner("Validating key and fetching models..."):
                    if st.session_state.provider == "OpenAI":
                        ids = validate_and_list_openai(api_key_input)
                    else:
                        ids = validate_and_list_anthropic(api_key_input)
                    rows = build_model_rows(provider_key, ids)
                st.session_state.models = rows
                st.session_state.validated = True
                if rows:
                    st.session_state.selected_model = rows[0]["id"]
            except Exception as e:
                st.session_state.validated = False
                st.session_state.models = []
                st.session_state.validation_error = f"Validation failed: {e}"

    if st.session_state.validation_error:
        st.error(st.session_state.validation_error)

    if st.session_state.validated and st.session_state.models:
        st.success(f"Key validated — {len(st.session_state.models)} model(s) found.")

        model_ids = [r["id"] for r in st.session_state.models]
        current = st.session_state.selected_model
        idx = model_ids.index(current) if current in model_ids else 0
        chosen = st.selectbox("Model to chat with", model_ids, index=idx)
        st.session_state.selected_model = chosen

        st.markdown("**Available models & pricing** (USD / 1M tokens)")
        table_md = "| Model | Input | Output |\n|---|---:|---:|\n"
        for r in st.session_state.models:
            in_p = f"${r['input']:.2f}" if r["input"] is not None else "—"
            out_p = f"${r['output']:.2f}" if r["output"] is not None else "—"
            marker = "**→ ** " if r["id"] == chosen else ""
            table_md += f"| {marker}{r['id']} | {in_p} | {out_p} |\n"
        st.markdown(table_md)
        st.caption(
            "Prices are a hand-maintained reference (checked Aug 2026) — "
            "the provider APIs don't return live pricing. Verify against "
            "the official pricing page for anything cost-critical."
        )

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — chat window
# ---------------------------------------------------------------------------
st.title("💡 Concept Explainer")
st.caption(
    "Enter a concept, or a handful of keywords, and get one explanation "
    "written for everyone in the room — plain-English analogies, the "
    "domain-expert framing, the technical deep dive, and the business "
    "impact. Then keep asking follow-ups in the same thread."
)

if not st.session_state.validated:
    st.info("👈 Enter and validate an API key in the sidebar to get started.")

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

placeholder = (
    "e.g. \"NLP\"  or  \"vector databases, embeddings, RAG\"  or a follow-up question..."
)
user_input = st.chat_input(placeholder, disabled=not st.session_state.validated)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = call_model(st.session_state.messages)
            except Exception as e:
                reply = f"⚠️ Something went wrong calling the model: {e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------------------------
# Download conversation as .txt
# ---------------------------------------------------------------------------
if st.session_state.messages:
    lines = [
        f"Concept Explainer conversation — exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Provider: {st.session_state.provider} | Model: {st.session_state.selected_model}",
        "=" * 70,
        "",
    ]
    for msg in st.session_state.messages:
        speaker = "YOU" if msg["role"] == "user" else "ASSISTANT"
        lines.append(f"[{speaker}]\n{msg['content']}\n")
    transcript = "\n".join(lines)

    st.download_button(
        label="⬇️ Download conversation (.txt)",
        data=io.BytesIO(transcript.encode("utf-8")),
        file_name=f"concept_explainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
