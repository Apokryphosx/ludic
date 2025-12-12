from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Any

from ludic.agents.tool_agent import ToolAgent
from ludic.types import SamplingArgs
from ludic.parsers import ParseResult

class ReActAgent(ToolAgent):
    """
    An agent that implements the ReAct pattern: 
    [Think -> Tool Call] * n -> Act.

    It supports an execution loop where the model can call auxiliary tools 
    multiple times before emitting a final answer for the environment.

    If the max_react_steps limit is reached, it forces a final generation 
    attempt without tools to produce a valid environment action.
    """
    def __init__(self, tools: List[Callable], max_react_steps: int = 5, **kwargs):
        """
        Args:
            tools: List of python functions the agent can call.
            max_react_steps: Maximum number of internal think/tool loops.
            **kwargs: Passed to base Agent.
        """
        super().__init__(tools=tools, **kwargs)
        self.max_react_steps = max_react_steps
        
        # Safety check: Context must explicitly flag support for tools
        if not self._ctx.supports_tools:
            raise TypeError("ReActAgent requires a context with supports_tools=True. ")

    async def act(
        self, 
        sampling_args: SamplingArgs, 
        timeout_s: float | None = None
    ) -> Tuple[ParseResult, str, Dict[str, Any]]:
        
        # 1. Setup Sampling Args
        sargs = self._with_tools(sampling_args)
        last_info: Dict[str, Any] = {}
        
        # 2. ReAct Loop
        for step_i in range(self.max_react_steps):
            
            # --- THE SHOT CLOCK LOGIC ---
            # If we are on the last step, force a final answer.
            is_final_try = (step_i == self.max_react_steps - 1)
            
            messages = self._ctx.on_before_act()

            if is_final_try:
                # A. Strip tools so it CANNOT call them
                sargs = self._strip_tools(sargs)
                
                # B. Force the agent to wrap up
                # We inject a temporary "system" instruction into the prompt
                # (Note: We don't save this to self._ctx, just for this one call)
                messages = messages + [{
                    "role": "user", 
                    "content": (
                        "You have exhausted your reasoning steps. "
                        "You must output your final move now."
                    )
                }]

            # 3. Inference (shared helper)
            resp, info, last_info = await self._infer_once(
                messages=messages,
                sampling_args=sargs,
                timeout_s=timeout_s,
            )

            # Extract content/tool_calls from OpenAI raw_response
            content, tool_calls = self._extract_openai_message(info)

            # 4. Handle Final Panic Move
            if is_final_try:
                # We forced it to output text. Return whatever it gave us.
                final_text = content or ""
                parse_result = self._parser(final_text)
                
                # Update context so the agent remembers its final decision
                # We pass None for tool_calls because tools were disabled
                self._ctx.add_assistant_step(final_text, None)
                
                return parse_result, final_text, last_info

            # 5. Normal Logic (Update Context & Check Tools)
            self._ctx.add_assistant_step(content, tool_calls)

            if tool_calls:
                # --- EXECUTION PHASE ---
                self._run_tool_calls(tool_calls)
                
                # Continue loop -> Model sees result and thinks again
                continue
            
            else:
                # --- FINAL ANSWER PHASE ---
                # No tool calls means the model is talking to the user/env.
                # We expect the content to contain the action here.
                # The format is defined by self._parser (XML, JSON, Regex, etc).
                final_text = content or ""
                parse_result = self._parser(final_text)
                return parse_result, final_text, last_info

        # This should technically be unreachable due to the is_final_try block,
        # but good for safety.
        fallback_msg = "Error: Loop logic failure."
        return self._parser(fallback_msg), fallback_msg, last_info
