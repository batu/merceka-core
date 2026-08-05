# robustness: wa_bot webhook POST has no signature verification and no body-size cap

`merceka_core/wa_bot/webhook.py` verifies `verify_token` only on the GET
verification handshake (~line 261). The POST handler `webhook_receive` (~270-315)
does **no** `X-Hub-Signature-256` HMAC check — the module's own docstring flags it
as a TODO at line 239 — and calls `await req.body()` (~line 289) with no
`Content-Length`/size guard before `json.loads`. Each accepted message invokes the
caller-supplied `handler` (~line 308), which for this library's purpose typically
triggers an LLM call.

**Blast radius:** anyone who discovers the webhook URL can POST arbitrary JSON with
no proof of Meta origin, turning an unauthenticated public endpoint into
attacker-driven LLM spend (cost amplification) and, via the unbounded body read, a
trivial large-payload memory-exhaustion DoS. This is internet-facing; small scale
does not shrink the blast radius of "anyone can run up your OpenRouter bill."

## Decided approach
- **Signature:** verify `X-Hub-Signature-256` as HMAC-SHA256 of the **raw** request
  body keyed by the Meta **app secret**, using `hmac.compare_digest`. Read the secret
  from config (extend `wa_bot/config.py` — it already holds WhatsApp creds like
  `waba_id`/`verify_token`; add `app_secret`, sourced from env, not hardcoded).
  Reject (HTTP 403) when the header is missing or the digest mismatches. Compute the
  HMAC over the exact bytes returned by `req.body()` **before** `json.loads` (signature
  is over raw bytes). If no `app_secret` is configured, fail closed with a clear
  startup/first-request error rather than silently accepting unsigned traffic.
- **Body cap:** enforce a max body size (e.g. reject if `Content-Length` exceeds a
  small constant, and cap the actual read) before parsing. WhatsApp webhook payloads
  are small; a few hundred KB ceiling is generous.

Rejected: a full middleware/auth framework — a ~15-line HMAC check plus a size guard
is proportionate. Rejected: making signature optional via a flag — fail closed.

## Scope fence
- `merceka_core/wa_bot/webhook.py` (POST handler: add signature check + size guard).
- `merceka_core/wa_bot/config.py` (add `app_secret` config field, env-sourced).
- `tests/wa_bot/test_webhook.py` (+ `tests/wa_bot/conftest.py` fixtures as needed):
  add valid-signature-accepted, bad/missing-signature-rejected, oversize-rejected cases.
`wa_bot` is behind the `wa-bot` extra — do not pull FastHTML into base imports.

## Acceptance criteria
- POST with a valid `X-Hub-Signature-256` over the body → handler runs.
- POST with missing or wrong signature → 403, handler NOT called.
- POST exceeding the size cap → rejected before `json.loads`.
- Existing GET verify-token handshake behavior unchanged.
- No secret is hardcoded; `app_secret` comes from config/env.

## Verify (baseline stamped 2026-07-06, main @ 1e859de)
```
cd /home/batu/Desktop/utolye/merceka_core
uv run pytest tests/ -q                # baseline: 346 passed, 1 skipped, 6 deselected
uv run pytest tests/wa_bot/ -q         # exercises the webhook/client/config surface
uv run pytest tests/contracts/ -q      # baseline: 17 passed  (consumer-import gate)
```

## Constraints
No PRs; conductor merges. Do not commit any real app secret. Out-of-fence needs →
handoff SURPRISES.
