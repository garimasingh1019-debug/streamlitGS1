# Concept Explainer

A Streamlit app where you bring your own OpenAI or Anthropic (Claude) API
key, browse the available models with pricing, and get a multi-audience
explanation (plain English + analogy, domain-expert framing, technical
deep dive, business impact) of any concept — then keep asking follow-ups
in the same chat thread and download the transcript.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` (and this README) to a GitHub repo.
2. Go to https://share.streamlit.io , sign in, and click "New app".
3. Point it at your repo/branch and set the main file to `app.py`.
4. Deploy. No secrets/config needed — each visitor pastes their own API
   key into the sidebar at runtime; it lives only in that session's memory
   and is never written to disk, logged, or sent anywhere except directly
   to the provider's API.

## How it works

- **Validate & fetch models**: calls the provider's lightweight
  `models.list()` endpoint. That single call both confirms the key works
  and returns every model the key can access — no wasted generation
  tokens spent just to "test" the key.
- **Pricing table**: neither provider's API returns live pricing, so
  `PRICING_USD_PER_1M` in `app.py` is a hand-maintained reference table
  (checked August 2026). It's matched by model id, with a prefix-based
  fallback for dated snapshots of a known model. Update this dict as
  providers change prices, or extend it with new model families —
  unmatched models just show `—` instead of breaking anything.
- **Explanations**: a system prompt instructs the model to always answer
  in five sections — *In Plain English*, *For Domain Experts/SMEs*, *For
  Technical Practitioners*, *Business Impact*, *Key Takeaways* — for the
  first message on a new concept, then answer follow-ups directly.
- **Download**: the full chat transcript (your prompts + the model's
  answers) can be exported as a timestamped `.txt` file at any time.

## Notes & possible tweaks

- If a very new OpenAI reasoning-model family requires the Responses API
  instead of Chat Completions, swap `call_openai()` to use
  `client.responses.create(...)`.
- The OpenAI model list is filtered to chat-oriented prefixes
  (`gpt-`, `o1`, `o3`, `o4`, `chatgpt`) to avoid cluttering the dropdown
  with embedding/moderation/image models — adjust `preferred_prefixes` in
  `validate_and_list_openai()` if you want everything.
- Switching provider in the sidebar resets the model list/validation but
  keeps your chat history.
