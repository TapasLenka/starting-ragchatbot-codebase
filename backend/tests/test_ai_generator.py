import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from ai_generator import AIGenerator


# ── Helpers to build fake Anthropic SDK response objects ──────────────────────

def make_tool_use_response(
    tool_name="search_course_content",
    tool_input=None,
    tool_id="toolu_01TESTID",
):
    if tool_input is None:
        tool_input = {"query": "test query"}
    tool_block = SimpleNamespace(type="tool_use", id=tool_id, name=tool_name, input=tool_input)
    return SimpleNamespace(stop_reason="tool_use", content=[tool_block])


def make_text_response(text="Here is your answer."):
    text_block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(stop_reason="end_turn", content=[text_block])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_anthropic_client():
    return MagicMock()


@pytest.fixture
def ai_generator(mock_anthropic_client):
    with patch("ai_generator.anthropic.Anthropic", return_value=mock_anthropic_client):
        gen = AIGenerator(api_key="test-key", model="claude-test")
    gen.client = mock_anthropic_client
    return gen


@pytest.fixture
def mock_tool_manager():
    manager = MagicMock()
    manager.execute_tool.return_value = "Lesson 1 content: RAG is retrieval augmented generation."
    return manager


@pytest.fixture
def dummy_tools():
    return [{"name": "search_course_content", "description": "search", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]


# ── Tests: direct (no-tool) response path ─────────────────────────────────────

class TestDirectResponse:

    def test_general_query_makes_exactly_one_api_call(self, ai_generator, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_text_response("Paris is the capital.")
        ai_generator.generate_response("What is the capital of France?")
        assert mock_anthropic_client.messages.create.call_count == 1

    def test_general_query_returns_text_content(self, ai_generator, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_text_response("Paris is the capital.")
        result = ai_generator.generate_response("What is the capital of France?")
        assert result == "Paris is the capital."

    def test_no_tool_call_when_no_tools_provided(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        mock_anthropic_client.messages.create.return_value = make_text_response("answer")
        ai_generator.generate_response("What is Python?", tool_manager=mock_tool_manager)
        mock_tool_manager.execute_tool.assert_not_called()


# ── Tests: sequential tool-execution loop ─────────────────────────────────────

class TestSequentialToolExecution:

    def test_no_tool_use_baseline(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        """With tools available, Claude answers immediately without invoking any tool."""
        mock_anthropic_client.messages.create.return_value = make_text_response("General answer.")
        result = ai_generator.generate_response(
            "What is 2+2?", tools=dummy_tools, tool_manager=mock_tool_manager
        )
        assert mock_anthropic_client.messages.create.call_count == 1
        mock_tool_manager.execute_tool.assert_not_called()
        assert result == "General answer."

    def test_single_round_claude_answers_directly_after_tool(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        """After 1 tool round Call 2 (tools still available) returns end_turn — no extra synthesis."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response("Answer after one search."),
        ]
        result = ai_generator.generate_response(
            "What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager
        )
        assert mock_anthropic_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once()
        assert result == "Answer after one search."

    def test_two_rounds_then_synthesis(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        """Two tool rounds exhaust MAX_TOOL_ROUNDS; a synthesis call (no tools) follows."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="tu_1"),
            make_tool_use_response(tool_id="tu_2"),
            make_text_response("Final synthesised answer."),
        ]
        result = ai_generator.generate_response(
            "What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager
        )
        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result == "Final synthesised answer."

    def test_tool_error_terminates_loop(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        """A tool execution error skips further rounds; the synthesis call has no tools."""
        mock_tool_manager.execute_tool.side_effect = RuntimeError("DB unavailable")
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="tu_err"),
            make_text_response("I encountered an error."),
        ]
        result = ai_generator.generate_response(
            "What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager
        )
        assert mock_anthropic_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once()

        synthesis_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" not in synthesis_kwargs

        tool_result_block = synthesis_kwargs["messages"][-1]["content"][0]
        assert "Tool error:" in tool_result_block["content"]
        assert result == "I encountered an error."

    def test_max_tool_rounds_enforced(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        """The inter-round API call has tools; the synthesis call (Call 3) must not."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="tu_1"),
            make_tool_use_response(tool_id="tu_2"),
            make_text_response("done"),
        ]
        ai_generator.generate_response(
            "What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager
        )
        calls = mock_anthropic_client.messages.create.call_args_list
        # Call 2 (inter-round, index 1) keeps tools so Claude can use them again
        assert "tools" in calls[1][1]
        # Call 3 (synthesis, index 2) must strip tools to prevent further looping
        assert "tools" not in calls[2][1]
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_synthesis_message_list_has_five_entries_after_two_rounds(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        """Synthesis call receives exactly 5 messages:
        [user_query, asst_round1, result_round1, asst_round2, result_round2]."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="tu_1"),
            make_tool_use_response(tool_id="tu_2"),
            make_text_response("done"),
        ]
        ai_generator.generate_response(
            "What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager
        )
        synthesis_kwargs = mock_anthropic_client.messages.create.call_args_list[2][1]
        messages = synthesis_kwargs["messages"]
        assert len(messages) == 5, f"Expected 5 messages, got {len(messages)}: {messages}"
