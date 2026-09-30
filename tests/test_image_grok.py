"""grok/ image models, run through a fake ``grok`` CLI placed on PATH.

The fake mirrors the real CLI's contract: it reads ``--prompt-file`` and
``--cwd``, writes the image under ``<cwd>/output/``, and prints one JSON result
with ``usage``, ``total_cost_usd``, ``sessionId`` and ``num_turns``.
FAKE_GROK_MODE picks a failure mode.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from merceka_core import costs
from merceka_core import image as image_module
from merceka_core.image import generate_image

CLI_SESSION = "01a0a9d0-d40b-77c2-b646-911373e8f29f"
FAKE_GROK = textwrap.dedent(
  f"""\
  #!{sys.executable}
  import json, os, re, sys, time
  from PIL import Image

  args = sys.argv[1:]
  job_dir = args[args.index("--cwd") + 1]
  prompt = open(args[args.index("--prompt-file") + 1]).read()
  with open(os.environ["FAKE_GROK_PROMPT_LOG"], "a") as fh:
    fh.write(prompt)
  source = re.search(r"absolute path (\\S+) ", prompt)
  if source and not os.path.exists(source.group(1)):
    print("source image missing", file=sys.stderr)
    sys.exit(3)
  mode = os.environ.get("FAKE_GROK_MODE", "ok")
  if mode == "sleep":
    time.sleep(30)
  if mode == "fail":
    print("auth expired", file=sys.stderr)
    sys.exit(2)
  if mode == "ok":
    Image.new("RGB", (64, 64), (1, 2, 3)).save(os.path.join(job_dir, "output", "result.png"))
  print(json.dumps({{
    "text": "output/result.png",
    "stopReason": "end_turn",
    "usage": {{"input_tokens": 100, "output_tokens": 20}},
    "total_cost_usd": 0.02,
    "sessionId": "{CLI_SESSION}",
    "num_turns": 3,
  }}))
  """
)


@pytest.fixture
def grok(tmp_path, monkeypatch):
  bin_dir = tmp_path / "bin"
  bin_dir.mkdir()
  cli = bin_dir / "grok"
  cli.write_text(FAKE_GROK)
  cli.chmod(0o755)
  monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
  monkeypatch.setenv("FAKE_GROK_PROMPT_LOG", str(tmp_path / "prompts.log"))
  scratch = tmp_path / "scratch"
  scratch.mkdir()
  monkeypatch.setattr(tempfile, "tempdir", str(scratch))

  def set_mode(mode: str) -> Path:
    monkeypatch.setenv("FAKE_GROK_MODE", mode)
    return scratch / "merceka-grok"

  return set_mode


def _ledger_rows() -> list[dict]:
  path = costs.ledger_path()
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines()]


def _job_dirs(base: Path) -> list[Path]:
  return list(base.iterdir()) if base.exists() else []


def test_generation_returns_the_cli_image_and_records_its_cost(grok):
  grok("ok")

  result = generate_image("a tree", model="grok/imagine")

  assert result.size == (64, 64)
  [row] = _ledger_rows()
  assert row["source"] == "grok-cli"
  assert row["model"] == "grok/imagine"
  assert row["usd"] == 0.02
  assert row["usage"] == {"input_tokens": 100, "output_tokens": 20}


def test_cli_session_id_does_not_replace_the_ambient_session_id(grok):
  grok("ok")

  with costs.attribution({"app": "ftb-level-editor", "sessionId": "level_abc"}):
    generate_image("a tree", model="grok/imagine")

  [row] = _ledger_rows()
  assert row["meta"]["sessionId"] == "level_abc"
  assert row["meta"]["grokSessionId"] == CLI_SESSION
  assert row["meta"]["numTurns"] == 3


def test_billed_run_without_an_image_is_still_recorded(grok):
  grok("no_image")

  with pytest.raises(RuntimeError, match="grok produced no image"):
    generate_image("a tree", model="grok/imagine")

  [row] = _ledger_rows()
  assert row["usd"] == 0.02
  assert row["meta"]["status"] == "no_image"


def test_timed_out_run_is_recorded_unpriced(grok, monkeypatch):
  grok("sleep")
  monkeypatch.setattr(image_module, "_GROK_TIMEOUT_S", 1)

  with pytest.raises(subprocess.TimeoutExpired):
    generate_image("a tree", model="grok/imagine")

  [row] = _ledger_rows()
  assert row["usd"] is None
  assert row["meta"] == {"status": "timeout"}


def test_failed_run_is_recorded(grok):
  grok("fail")

  with pytest.raises(RuntimeError, match="grok CLI failed"):
    generate_image("a tree", model="grok/imagine")

  [row] = _ledger_rows()
  assert row["meta"] == {"status": "error"}
