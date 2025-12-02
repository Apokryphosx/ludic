from __future__ import annotations
from typing import Optional

from ludic.env import Env
from ludic.agent import Agent
from ludic.types import Rollout, Step, StepOutcome, SamplingArgs
from .base import InteractionProtocol

class SingleAgentSyncProtocol(InteractionProtocol):
    """
    Implements the standard single-agent, synchronous interaction loop.
    
    This protocol is initialized with a single, fully-configured Agent
    that manages its own context and parsing.
    """
    
    def __init__(self, agent: Agent):
        """
        Initializes the protocol with the single agent it will use.
        
        Args:
            agent: A fully-configured Agent instance.
        """
        self.agent = agent

    async def run(
        self,
        *,
        env: Env,
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
    ) -> Rollout:
        
        # 1. --- Setup ---
        agent = self.agent  # Use the agent provided during initialization
        sargs: SamplingArgs = sampling_args or {}

        # 2. --- Reset Env ---
        obs, info = env.reset(seed=seed)
        
        # 3. --- Reset Agent & Feed First Obs ---
        # Pass the env's suggested prompt to the agent.
        # The agent's ContextStrategy will decide whether to use
        # this, or its own default prompt.
        agent.reset(system_prompt=env.suggested_sysprompt)
        
        # Feed the *first* observation to the agent
        agent.on_env_reset(obs, info) 
        
        rollout = Rollout(meta={
            "agent_name": getattr(agent, "name", "unknown"),
            "env_name": env.__class__.__name__,
        })

        # 4. --- Run Interaction Loop ---
        for t in range(max_steps):
            
            # Store the obs that led to this action (for logging)
            current_obs_for_step = obs 
            
            # --- A. Call the Agent ---
            # Agent acts based on its internal context (fed by previous obs)
            parse_result, raw_action, client_info = await agent.act(
                sampling_args=sargs,
                timeout_s=timeout_s
            )

            # --- B. Handle Parser Failure ---
            if parse_result.action is None:
                synthetic_obs = parse_result.obs or "Invalid action."
                parser_reward = parse_result.reward

                rollout.steps.append(Step(
                    index=t,
                    prev_obs=current_obs_for_step,
                    action=raw_action,
                    next_obs=synthetic_obs,
                    reward=parser_reward,
                    truncated=False,
                    terminated=False,
                    info={
                        "parse_error": True,
                        "raw_action": raw_action,
                        **client_info
                    },
                ))

                # Feed the synthetic failure obs back to the agent
                obs = synthetic_obs
                info = {"parse_error": True}
                agent.on_after_step(obs, info)
                continue # Continue to the next loop iteration

            # --- C. Handle Parser Success (Step Env) ---
            parsed_action = parse_result.action
            parser_reward = parse_result.reward

            outcome: StepOutcome = env.step(parsed_action)

            # Build info dict
            step_info = {
                **client_info,
                **outcome.info,
                "parsed_action": parsed_action,
            }

            # Total reward = env reward + parser reward
            total_reward = outcome.reward + parser_reward

            # For logging: terminal/truncated steps have no next_obs
            logged_next_obs = None
            if not (outcome.terminated or outcome.truncated):
                logged_next_obs = outcome.obs

            rollout.steps.append(Step(
                index=t,
                prev_obs=current_obs_for_step,
                action=raw_action,
                next_obs=logged_next_obs,
                reward=total_reward,
                truncated=outcome.truncated,
                terminated=outcome.terminated,
                info=step_info,
            ))

            # Update obs and info for the next loop
            obs = outcome.obs
            info = outcome.info

            # --- D. Check for Termination & Feed Next Obs ---
            if outcome.terminated or outcome.truncated:
                break # Exit loop
            else:
                # Feed the new observation to the agent for the *next* step
                agent.on_after_step(obs, info)

        return rollout