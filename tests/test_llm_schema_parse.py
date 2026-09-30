"""LLM._parse_response rejects JSON that shares no field with the output schema.

Regression (review R17): a schema whose fields all have defaults accepted an
envelope such as {"result": {...}} and returned an all-default object, so the
answer was silently lost. It now raises pydantic's ValidationError, the error a
strict schema already raises for the same response, and the one callers such
as the fabrikav2 level editor already catch and retry.
"""

import pytest
from pydantic import ValidationError

from merceka_core.llm import LLM, OutputSchema


class Verdict(OutputSchema):
  issues: list[str] = []
  approved: bool = False


@pytest.fixture
def llm():
  return LLM("openrouter/x", output_schema=Verdict)


def test_envelope_json_raises_instead_of_returning_defaults(llm):
  with pytest.raises(ValidationError, match="none of the schema's fields"):
    llm._parse_response('{"result": {"issues": ["hole in wing"], "approved": true}}')


def test_envelope_dict_raises(llm):
  with pytest.raises(ValidationError, match="none of the schema's fields"):
    llm._parse_response({"result": {"approved": True}})


def test_matching_json_still_parses(llm):
  out = llm._parse_response('{"issues": ["hole"], "approved": true, "extra": 1}')
  assert out.issues == ["hole"] and out.approved is True


def test_partial_json_uses_defaults_for_the_rest(llm):
  assert llm._parse_response('{"approved": true}') == Verdict(approved=True)


def test_empty_object_is_still_accepted(llm):
  """No keys at all is not an envelope; the defaults are the answer."""
  assert llm._parse_response("{}") == Verdict()


def test_content_field_counts_as_a_schema_field(llm):
  assert llm._parse_response('{"content": "looks fine"}').content == "looks fine"


def test_aliased_field_counts_as_a_schema_field():
  from pydantic import Field

  class Aliased(OutputSchema):
    is_ok: bool = Field(default=False, alias="isOk")

  assert LLM("openrouter/x", output_schema=Aliased)._parse_response('{"isOk": true}').is_ok


def test_root_model_keys_are_data_not_fields():
  from pydantic import RootModel

  class Scores(RootModel[dict[str, int]]):
    pass

  out = LLM("openrouter/x", output_schema=Scores)._parse_response('{"a": 1, "b": 2}')
  assert out.root == {"a": 1, "b": 2}
