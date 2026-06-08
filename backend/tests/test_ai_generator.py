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


# ── Tests: tool-execution path ────────────────────────────────────────────────

class TestToolExecution:

    def test_tool_use_response_triggers_two_api_calls(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response("RAG stands for Retrieval Augmented Generation."),
        ]
        ai_generator.generate_response("What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager)
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_final_text_is_from_second_response(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response("final answer from second call"),
        ]
        result = ai_generator.generate_response("What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager)
        assert result == "final answer from second call"

    def test_tool_manager_execute_called_with_correct_args(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_name="search_course_content", tool_input={"query": "RAG basics", "course_name": "AI101"}),
            make_text_response("answer"),
        ]
        ai_generator.generate_response("What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager)
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="RAG basics", course_name="AI101"
        )

    def test_second_api_call_has_no_tools_parameter(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response("answer"),
        ]
        ai_generator.generate_response("What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager)

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
        assert "tool_choice" not in second_call_kwargs

    def test_second_call_message_list_has_three_entries(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        """Messages must be: [user_query, assistant_tool_use, user_tool_result]."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response("answer"),
        ]
        ai_generator.generate_response("What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager)

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}: {messages}"

    def test_tool_result_message_has_correct_structure(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        mock_tool_manager.execute_tool.return_value = "course content here"
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="toolu_CHECKID"),
            make_text_response("answer"),
        ]
        ai_generator.generate_response("What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager)

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        tool_result_msg = second_call_kwargs["messages"][-1]

        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        result_block = tool_result_msg["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "toolu_CHECKID"
        assert result_block["content"] == "course content here"

    def test_assistant_message_contains_original_response_content(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        first_response = make_tool_use_response()
        mock_anthropic_client.messages.create.side_effect = [
            first_response,
            make_text_response("answer"),
        ]
        ai_generator.generate_response("What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager)

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assistant_msg = second_call_kwargs["messages"][1]
        assert assistant_msg["role"] == "assistant"
        # The SDK content list from the first response must be passed through verbatim
        assert assistant_msg["content"] is first_response.content

    def test_tool_returning_error_string_does_not_raise(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, dummy_tools
    ):
        mock_tool_manager.execute_tool.return_value = "Search error: n_results exceeds collection count"
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response("I could not find that information."),
        ]
        # Must NOT raise — the error string is just passed as context to Claude
        result = ai_generator.generate_response("What is RAG?", tools=dummy_tools, tool_manager=mock_tool_manager)
        assert isinstance(result, str)
