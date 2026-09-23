# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for max_whitespace_cnt parameter passing to xgrammar."""

import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from vllm.config import (
    ModelConfig,
    SchedulerConfig,
    StructuredOutputsConfig,
    VllmConfig,
)
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    _with_max_whitespace_cnt,
)

pytestmark = pytest.mark.cpu_test


class TestMaxWhitespaceCnt:
    """Test that max_whitespace_cnt is passed correctly to xgrammar."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock VllmConfig with custom structured outputs config."""
        model_config = Mock(spec=ModelConfig)
        model_config.skip_tokenizer_init = True
        model_config.get_vocab_size = Mock(return_value=50000)
        model_config.runner_type = "generate"
        model_config.tokenizer = "test-tokenizer"
        model_config.tokenizer_mode = "auto"
        model_config.trust_remote_code = False
        model_config.tokenizer_revision = None

        scheduler_config = Mock(spec=SchedulerConfig)
        scheduler_config.max_num_seqs = 128

        config = Mock(spec=VllmConfig)
        config.model_config = model_config
        config.scheduler_config = scheduler_config
        config.speculative_config = None
        return config

    def _make_backend(self, mock_vllm_config, max_whitespace_cnt):
        """Helper to create an XgrammarBackend with given max_whitespace_cnt."""
        mock_vllm_config.structured_outputs_config = StructuredOutputsConfig(
            backend="xgrammar",
            max_whitespace_cnt=max_whitespace_cnt,
        )
        return XgrammarBackend(
            vllm_config=mock_vllm_config,
            tokenizer=Mock(),
            vocab_size=50000,
        )

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_max_whitespace_cnt_default(self, mock_xgr, mock_vllm_config):
        """Verify default max_whitespace_cnt=8 is passed to compile_json_schema."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        backend = self._make_backend(mock_vllm_config, max_whitespace_cnt=8)

        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        kwargs = call.call_args.kwargs
        assert kwargs.get("max_whitespace_cnt") == 8
        assert kwargs.get("any_whitespace") is True

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_max_whitespace_cnt_one(self, mock_xgr, mock_vllm_config):
        """Verify max_whitespace_cnt=1 is passed correctly."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        backend = self._make_backend(mock_vllm_config, max_whitespace_cnt=1)

        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        assert call.call_args.kwargs.get("max_whitespace_cnt") == 1

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_max_whitespace_cnt_none(self, mock_xgr, mock_vllm_config):
        """Verify max_whitespace_cnt=None (unbounded) is passed correctly."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        backend = self._make_backend(mock_vllm_config, max_whitespace_cnt=None)

        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        assert call.call_args.kwargs.get("max_whitespace_cnt") is None

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_max_whitespace_cnt_json_object(self, mock_xgr, mock_vllm_config):
        """Verify max_whitespace_cnt is passed for JSON object (no schema) case."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        backend = self._make_backend(mock_vllm_config, max_whitespace_cnt=2)

        backend.compile_grammar(StructuredOutputOptions.JSON_OBJECT, "")

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        assert call.call_args.kwargs.get("max_whitespace_cnt") == 2

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_disable_any_whitespace_overrides(self, mock_xgr, mock_vllm_config):
        """Verify disable_any_whitespace=True is syntactic sugar for
        any_whitespace=False + max_whitespace_cnt=0."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        mock_vllm_config.structured_outputs_config = StructuredOutputsConfig(
            backend="xgrammar",
            disable_any_whitespace=True,
            max_whitespace_cnt=2,  # should be overridden to 0
        )
        backend = XgrammarBackend(
            vllm_config=mock_vllm_config,
            tokenizer=Mock(),
            vocab_size=50000,
        )

        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        assert call.call_args.kwargs.get("any_whitespace") is False
        assert call.call_args.kwargs.get("max_whitespace_cnt") == 0


class TestXgrammarFSMWhitespace:
    """Verify xgrammar FSM behavior at the library level.

    These tests use real xgrammar (not mocks) to confirm that
    max_whitespace_cnt actually constrains whitespace in the FSM.
    """

    @pytest.fixture
    def tokenizer_info(self):
        """Create a minimal TokenizerInfo for FSM testing."""
        from xgrammar import TokenizerInfo, VocabType

        return TokenizerInfo(
            encoded_vocab=["{", "}", ":", '"', " ", "a", "1", "\n", "\t"],
            vocab_type=VocabType.RAW,
            vocab_size=9,
            stop_token_ids=[],
            add_prefix_space=False,
        )

    def _advance_to_value(self, matcher) -> None:
        """Advance FSM to the JSON value position (after the colon)."""
        for ch in ["{", '"', "a", '"', ":"]:
            assert matcher.accept_string(ch), f"Failed to accept {repr(ch)}"

    def test_fsm_limits_whitespace_with_max_cnt_2(self, tokenizer_info):
        """With max_whitespace_cnt=2, FSM rejects 3rd+ consecutive space."""
        from xgrammar import GrammarCompiler, GrammarMatcher

        compiler = GrammarCompiler(tokenizer_info)
        schema = (
            '{"type": "object", "properties": {"a": {"type": "integer"}}, '
            '"required": ["a"]}'
        )
        compiled = compiler.compile_json_schema(schema, max_whitespace_cnt=2)
        matcher = GrammarMatcher(compiled)

        self._advance_to_value(matcher)

        # First 2 spaces accepted, 3rd rejected
        assert matcher.accept_string(" "), "1st space should be accepted"
        assert matcher.accept_string(" "), "2nd space should be accepted"
        assert not matcher.accept_string(" "), "3rd space should be rejected"

    def test_fsm_allows_unlimited_whitespace_none(self, tokenizer_info):
        """With max_whitespace_cnt=None, FSM allows unlimited spaces."""
        from xgrammar import GrammarCompiler, GrammarMatcher

        compiler = GrammarCompiler(tokenizer_info)
        schema = (
            '{"type": "object", "properties": {"a": {"type": "integer"}}, '
            '"required": ["a"]}'
        )
        compiled = compiler.compile_json_schema(schema, max_whitespace_cnt=None)
        matcher = GrammarMatcher(compiled)

        self._advance_to_value(matcher)

        # All 5 spaces accepted (unbounded)
        for i in range(5):
            assert matcher.accept_string(" "), f"Space #{i + 1} should be accepted"

    def test_fsm_single_whitespace_limit(self, tokenizer_info):
        """With max_whitespace_cnt=1, FSM rejects 2nd+ consecutive space."""
        from xgrammar import GrammarCompiler, GrammarMatcher

        compiler = GrammarCompiler(tokenizer_info)
        schema = (
            '{"type": "object", "properties": {"a": {"type": "integer"}}, '
            '"required": ["a"]}'
        )
        compiled = compiler.compile_json_schema(schema, max_whitespace_cnt=1)
        matcher = GrammarMatcher(compiled)

        self._advance_to_value(matcher)

        assert matcher.accept_string(" "), "1st space should be accepted"
        assert not matcher.accept_string(" "), "2nd space should be rejected"

    def test_fsm_allows_whitespace_then_accepts_value(self, tokenizer_info):
        """After limited whitespace, FSM still accepts the actual value."""
        from xgrammar import GrammarCompiler, GrammarMatcher

        compiler = GrammarCompiler(tokenizer_info)
        schema = (
            '{"type": "object", "properties": {"a": {"type": "integer"}}, '
            '"required": ["a"]}'
        )
        compiled = compiler.compile_json_schema(schema, max_whitespace_cnt=2)
        matcher = GrammarMatcher(compiled)

        self._advance_to_value(matcher)

        # 2 spaces accepted, then value "1" accepted
        assert matcher.accept_string(" ")
        assert matcher.accept_string(" ")
        assert not matcher.accept_string(" "), "3rd space should be rejected"
        assert matcher.accept_string("1"), "Value digit should be accepted"

    def test_fsm_string_value_whitespace_unaffected(self, tokenizer_info):
        """max_whitespace_cnt does NOT affect whitespace inside string values.

        The limit only applies between JSON structural tokens ({, }, :, ,, etc.).
        Spaces inside quoted string values should remain unlimited.
        """
        from xgrammar import GrammarCompiler, GrammarMatcher

        compiler = GrammarCompiler(tokenizer_info)
        schema = (
            '{"type": "object", "properties": {"a": {"type": "string"}}, '
            '"required": ["a"]}'
        )
        compiled = compiler.compile_json_schema(schema, max_whitespace_cnt=2)
        matcher = GrammarMatcher(compiled)

        # Advance to string value position: {"a":"
        for ch in ["{", '"', "a", '"', ":", '"']:
            assert matcher.accept_string(ch), f"Failed to accept {repr(ch)}"

        # Inside string value: unlimited spaces should be allowed
        for i in range(5):
            assert matcher.accept_string(" "), (
                f"Space #{i + 1} inside string value should be accepted"
            )


def _tool_call_tag(max_whitespace_cnt: int | None = None) -> dict[str, Any]:
    """Build a structural tag shaped like a strict tool call."""
    schema_node: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        },
    }
    if max_whitespace_cnt is not None:
        schema_node["max_whitespace_cnt"] = max_whitespace_cnt
    return {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": "<tool>",
                    "content": schema_node,
                    "end": "</tool>",
                }
            ],
        },
    }


def _tool_call_matcher(spec: str):
    """Compile a structural tag against a vocab holding the tool-call tokens."""
    from xgrammar import GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType

    tokenizer_info = TokenizerInfo(
        encoded_vocab=["{", "}", ":", '"', " ", "a", "1", "<tool>", "</tool>"],
        vocab_type=VocabType.RAW,
        vocab_size=9,
        stop_token_ids=[],
        add_prefix_space=False,
    )
    compiled = GrammarCompiler(tokenizer_info).compile_structural_tag(spec)
    matcher = GrammarMatcher(compiled)
    for ch in ["<tool>", "{", '"', "a", '"', ":"]:
        assert matcher.accept_string(ch), f"Failed to accept {repr(ch)}"
    return matcher


class TestStructuralTagMaxWhitespaceCnt:
    """Whitespace bounding on the structural tag (strict tool calling) path.

    xgrammar's compile_structural_tag takes no max_whitespace_cnt argument, so
    the bound has to be written onto each json_schema node of the tag instead.
    """

    def test_applies_bound_to_nested_json_schema_node(self):
        """The bound reaches a json_schema node nested inside a tag."""
        spec = json.dumps(_tool_call_tag())

        result = json.loads(_with_max_whitespace_cnt(spec, 4))

        assert result["format"]["tags"][0]["content"]["max_whitespace_cnt"] == 4

    def test_preserves_explicit_max_whitespace_cnt(self):
        """A bound already set by the caller wins over the config default."""
        spec = json.dumps(_tool_call_tag(max_whitespace_cnt=1))

        result = json.loads(_with_max_whitespace_cnt(spec, 8))

        assert result["format"]["tags"][0]["content"]["max_whitespace_cnt"] == 1

    def test_leaves_qwen_xml_parameter_nodes_alone(self):
        """qwen_xml_parameter carries a schema but has no whitespace field."""
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "qwen_xml_parameter",
                "json_schema": {"type": "string"},
            },
        }

        result = json.loads(_with_max_whitespace_cnt(json.dumps(tag), 4))

        assert "max_whitespace_cnt" not in result["format"]

    def test_does_not_touch_user_schema_contents(self):
        """Keys inside the JSON Schema document itself are left untouched."""
        tag = _tool_call_tag()
        tag["format"]["tags"][0]["content"]["json_schema"] = {
            "type": "object",
            "properties": {"nested": {"type": "json_schema"}},
        }

        result = json.loads(_with_max_whitespace_cnt(json.dumps(tag), 4))

        schema = result["format"]["tags"][0]["content"]["json_schema"]
        assert "max_whitespace_cnt" not in schema["properties"]["nested"]

    def test_does_not_mutate_input(self):
        """The walk returns new objects rather than editing the parsed tag."""
        tag = _tool_call_tag()
        spec = json.dumps(tag)

        _with_max_whitespace_cnt(spec, 4)

        assert json.loads(spec) == tag

    def test_returns_spec_unchanged_when_cnt_is_none(self):
        """None means unbounded, so the spec is passed through untouched."""
        spec = json.dumps(_tool_call_tag())

        assert _with_max_whitespace_cnt(spec, None) is spec

    @pytest.mark.parametrize("cnt", [0, -1])
    def test_returns_spec_unchanged_when_cnt_not_positive(self, cnt):
        """Non-positive bounds are rejected by xgrammar on structural tags."""
        spec = json.dumps(_tool_call_tag())

        assert _with_max_whitespace_cnt(spec, cnt) is spec

    def test_fsm_rejects_whitespace_beyond_bound(self):
        """End-to-end: the compiled tool-call grammar stops accepting spaces."""
        spec = _with_max_whitespace_cnt(json.dumps(_tool_call_tag()), 2)

        matcher = _tool_call_matcher(spec)

        assert matcher.accept_string(" "), "1st space should be accepted"
        assert matcher.accept_string(" "), "2nd space should be accepted"
        assert not matcher.accept_string(" "), "3rd space should be rejected"
        assert matcher.accept_string("1"), "Value digit should still be accepted"

    def test_fsm_unbounded_without_the_config(self):
        """Guards the test above: unbounded is the behavior being fixed."""
        matcher = _tool_call_matcher(json.dumps(_tool_call_tag()))

        for i in range(12):
            assert matcher.accept_string(" "), f"Space #{i + 1} should be accepted"
