# robustness: wa_bot client silently collapses transient and terminal failures to None

`merceka_core/wa_bot/client.py`'s `send_text` (~113-169), `send_template` (~171-254),
`get_media_url` (~256-298), and `download_media` (~300-335) each catch only
`httpx.RequestError` and otherwise return `None` on any `status_code >= 400`. So a
transient 429/5xx (retryable per this repo's own `retry.py` `_RETRY_STATUS_CODES`,
which is never imported here) is indistinguishable from a permanent 400 (bad
recipient/template): a user-facing reply is silently dropped on a transient blip, with
no retry and no signal to the caller that the failure was transient — a real bug for a
bot whose whole job is to reply. Secondarily, `resp.json()` on a 2xx-but-non-JSON body
is unguarded, so a malformed success response raises unhandled for any direct (non-
webhook) caller.

## Decided approach (narrowed per adversarial review — no framework)
Do **not** build a retry/circuit-breaker layer. Minimal change:
- Distinguish transient from terminal instead of collapsing both to `None`. Simplest
  proportionate option: raise a typed error (or return a tagged result) on failure so
  the caller can tell "network blip / 429 / 5xx → could retry" from "400 → give up".
  Reuse `retry.py`'s `_RETRY_STATUS_CODES` to classify rather than re-listing codes.
- Guard `resp.json()` so a non-JSON 2xx surfaces a clear error, not a raw
  `JSONDecodeError`.
- Keep the change behind the `wa-bot` extra; do not add new base dependencies.

Pick ONE of {raise typed error} or {return tagged result} and apply it consistently
across all four methods — do not mix. Prefer raising (it composes with the caller's own
try/except and makes silent loss impossible), unless the existing webhook call path
depends on the `None` sentinel — check `wa_bot/webhook.py` call sites first and adapt
them in the same change.

Rejected: a shared retry decorator around every client call — over-engineering at this
scale; the bug is the *loss of the transient/terminal distinction*, not the absence of
automatic retries. Rejected: silently retrying inside the client — surprises callers.

## Scope fence
- `merceka_core/wa_bot/client.py` (the four methods) and its webhook call sites in
  `merceka_core/wa_bot/webhook.py` if they consumed the `None` sentinel.
- `tests/wa_bot/test_client.py` (+ webhook tests if call sites change).

## Acceptance criteria
- A transient failure (mock 429/503) is distinguishable by the caller from a terminal
  400 — a test asserts they produce different, observable outcomes (not both `None`).
- A non-JSON 2xx body produces a clear typed error, not an unhandled `JSONDecodeError`.
- Webhook flow still works end-to-end (existing `tests/wa_bot/test_webhook.py` green).

## Verify (baseline stamped 2026-07-06, main @ 1e859de)
```
cd /home/batu/Desktop/utolye/merceka_core
uv run pytest tests/ -q          # baseline: 346 passed, 1 skipped, 6 deselected
uv run pytest tests/wa_bot/ -q   # wa_bot client + webhook + config + utils
```

## Constraints
No PRs; conductor merges. `wa_bot` is optional (`wa-bot` extra) — keep FastHTML out of
base imports. Out-of-fence → handoff SURPRISES.
