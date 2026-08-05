# idea: attach usage/cost to every LLM response (the data is already returned and discarded)

> Blue-sky card — brainstorm/plan before implementing; re-verify repo facts at pickup
> (baselines and line numbers drift). Confidence ~85%, complexity S.

## Summary
Surface token/cost usage on every `generate()`/`agenerate()` result. OpenRouter's
response body already contains a `usage` field (tokens, and cost when requested), and
Ollama's response object exposes token counts — but merceka_core reads only
`body["choices"][0]["message"]` and **discards `usage`** (verified: zero `usage` hits
in `llm.py`/`messages.py`; parsing in `_cloud_call_raw` llm.py:383-404 and
`_parse_openrouter_body` ~568). This is nearly free: parse data that already arrives.

## Why it matters (this scale)
Solo dev paying the OpenRouter bill personally across ~4 repos (mindweaver, slab,
videototext, wa_bot), with today the only spend signal being the OpenRouter dashboard
after the fact — disconnected from which call caused it. Attaching per-call usage
turns "what am I spending and where" into a local, join-able fact. Also unlocks the
lighter half of the killed caching/regression ideas: once you can see repeated-identical
spend, you'll know whether caching is worth it.

## Watch-outs
- Keep it additive and optional — do not change the default return type of
  `generate()` in a way that breaks the contract tests / consumer call sites (they
  expect `str | OutputSchema`). Prefer a side-channel (e.g. `llm.last_usage`, or an
  opt-in `return_usage=True`) over changing the primary return shape. This is a
  public-API surface — any change needs the `tests/contracts/` + import gate.
- Ollama and OpenRouter report usage differently; normalize to one small dataclass.
- Cost is only present when requested from OpenRouter; tokens are always available.
  Don't fabricate cost from a hardcoded price table that will rot.

## Complexity / confidence
S–M · ~85%. Highest value-to-cost item on the blue-sky list.
