import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- **Outline / syllabus / structure queries**: Use `get_course_outline` — never `search_course_content` for these.
- **Content / concept / lesson-detail queries**: Use `search_course_content`.
- **General knowledge questions**: Answer using existing knowledge without any tool call.
- **Sequential tool calls**: You may call tools up to two times per response when needed — for example, retrieve a course outline first to identify a lesson title, then search for related content using that title. After at most two tool calls, synthesise all retrieved information into a single complete answer.
- If a tool returns no results, state this clearly without offering alternatives.

Outline Response Format:
When `get_course_outline` is used, always include all of the following in your answer:
- Course title
- Course link (as a clickable markdown link)
- Each lesson: its number and title (e.g. "Lesson 1: Introduction")

Response Protocol:
- Provide direct answers only — no reasoning process, search explanations, or question-type analysis.
- Do not mention "based on the search results" or reference tool calls.

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _run_tool_loop(self, initial_response, base_params: Dict[str, Any], tool_manager) -> str:
        """
        Run up to MAX_TOOL_ROUNDS of tool execution, then make a final synthesis call.

        Each round:
          1. Appends the assistant tool-use response and tool results to the message list.
          2. If more rounds remain and no error occurred, makes the next API call WITH tools
             so Claude can choose to call another tool or answer directly.
          3. Terminates early if Claude answers without a tool call (condition B)
             or a tool execution raises (condition C).
        After all rounds, makes one unconditional synthesis call WITHOUT tools.

        Args:
            initial_response: First API response with stop_reason="tool_use"
            base_params: Original API parameters (includes messages, system, tools)
            tool_manager: Manager to execute tools

        Returns:
            Final response text
        """
        messages = base_params["messages"].copy()
        current_response = initial_response

        for round_idx in range(self.MAX_TOOL_ROUNDS):
            # Append assistant's tool-use content to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute every tool_use block in this response
            tool_results = []
            error_occurred = False
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                    except Exception as e:
                        result = f"Tool error: {str(e)}"
                        error_occurred = True

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

            # Stop looping on error (condition C) or after the last round (condition A)
            if error_occurred or round_idx + 1 >= self.MAX_TOOL_ROUNDS:
                break

            # More rounds available — call Claude again WITH tools so it can
            # choose to search again or answer directly
            next_response = self.client.messages.create(
                **self.base_params,
                messages=messages,
                system=base_params["system"],
                tools=base_params["tools"],
                tool_choice={"type": "auto"}
            )

            # Condition B: Claude chose to answer without another tool call
            if next_response.stop_reason != "tool_use":
                return next_response.content[0].text

            current_response = next_response

        # Unconditional final synthesis call — no tools so Claude cannot loop further
        final_response = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=base_params["system"]
        )
        return final_response.content[0].text
