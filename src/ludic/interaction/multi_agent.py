from __future__ import annotations
import asyncio
from typing import Optional, Dict, List

from ludic.envs.env import LudicEnv
from ludic.agents.base_agent import Agent
from ludic.types import Rollout, Step, StepOutcome, SamplingArgs
from .base import InteractionProtocol
from .step_collector import TraceCollector

class MultiAgentProtocol(InteractionProtocol):
    """
    Handles an interaction with multiple agents.
    
    This protocol queries the environment for which agents are
    'active' at each step and only gathers actions from them.
    
    It returns a list of Rollouts (one per managed agent) suitable
    for training.

    Parser failures:
      If a managed agent's parser returns ParseResult.action=None, that
      agent is omitted from env.step(actions). A synthetic Step is still
      logged for the failing agent (reward=parse_result.reward, info
      includes parse_error=True). The failing agent's context is updated
      with the synthetic observation for the next turn.
    """
    
    def __init__(self, agents: Dict[str, Agent]):
        """
        Initializes the protocol with a mapping of Agent IDs
        from the environment (e.g., "player_1") to the
        actual Agent objects that will fill those roles.
        """
        self.agent_map = agents

    async def run(
        self,
        *,
        env: LudicEnv,
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
    ) -> List[Rollout]:
        
        sargs: SamplingArgs = sampling_args or {}
        
        # Initialize the collector to track separate histories per agent
        collector = TraceCollector(
            protocol=self.__class__.__name__,
            env_name=env.__class__.__name__,
            agents=list(self.agent_map.keys())
        )

        # 1. --- Reset Env and all managed Agents ---
        obs_info_dict = env.reset(seed=seed)
        sys_prompt = getattr(env, "suggested_sysprompt", None)
        
        for agent_id, agent in self.agent_map.items():
            obs, info = obs_info_dict.get(agent_id, (None, {}))
            if obs is not None:
                agent.reset(system_prompt=sys_prompt)
                agent.on_env_reset(obs, info)
        
        # Track current observations for all managed agents
        current_obs = {k: v[0] for k, v in obs_info_dict.items()}

        # 2. --- Run Interaction Loop ---
        for t in range(max_steps):
            
            # --- A. Identify active agents ---
            active_ids = env.active_agents
            
            # Find which of the active agents are managed by this protocol
            agents_to_poll = {
                agent_id: self.agent_map[agent_id] 
                for agent_id in active_ids 
                if agent_id in self.agent_map
            }
            
            # --- B. Handle cases with no managed agents (Bot turns / Game Over) ---
            if not agents_to_poll:
                if not active_ids:
                    # No active agents at all implies the game is effectively over
                    break
                
                # Active agents exist but are not managed by us (e.g. internal bots/NPCs).
                # We must step the environment with an empty action set to let it proceed.
                outcomes_dict = env.step({})
                
                # Update observations for our agents (they might see what the bot did)
                for aid, out in outcomes_dict.items():
                    current_obs[aid] = out.obs
                    # We don't log a Step for our agents here (they didn't act),
                    # but we MUST update their context so they have the latest state.
                    if aid in self.agent_map:
                         self.agent_map[aid].on_after_step(out.obs, out.info)

                # Check if the bot turn ended the game
                if any(o.terminated or o.truncated for o in outcomes_dict.values()):
                    break
                
                continue 

            # --- C. Gather actions in parallel ---
            tasks = [
                agent.act(sampling_args=sargs, timeout_s=timeout_s)
                for agent in agents_to_poll.values()
            ]
            results = await asyncio.gather(*tasks)
            
            actions_to_take: Dict[str, str] = {}
            
            # Temporary storage to hold data needed for logging after the env step
            step_context_cache: Dict[str, Dict] = {} 
            
            for agent_id, (parse_result, raw, info) in zip(agents_to_poll.keys(), results):
                # Cache context for this specific agent's step
                step_context_cache[agent_id] = {
                    "prev_obs": current_obs.get(agent_id, ""),
                    "raw_action": raw,
                    "client_info": info,
                    "parse_result": parse_result,
                }
                
                if parse_result.action is not None:
                    actions_to_take[agent_id] = parse_result.action

            # --- D. Step the environment ---
            # actions_to_take only contains VALID actions from managed agents.
            # Invalid ones are excluded here but handled below via parser_failures.
            env_outcomes: Dict[str, StepOutcome] = {}
            if actions_to_take:
                env_outcomes = env.step(actions_to_take)
            # --- E. Process Results & Log Steps ---
            any_term = False
            any_trunc = False

            for agent_id, ctx in step_context_cache.items():
                parse_result = ctx["parse_result"]
                client_info = ctx["client_info"]
                
                # Did parsing fail?
                if parse_result.action is None:
                    # Create synthetic outcome for parser failure
                    synthetic_obs = parse_result.obs or "Invalid action."
                    outcome = StepOutcome(
                        obs=synthetic_obs,
                        reward=parse_result.reward,
                        truncated=False,
                        terminated=False,
                        info={"parse_error": True, **client_info}
                    )
                else:
                    # Normal environment outcome
                    # (It should exist in env_outcomes if we sent an action)
                    raw_outcome = env_outcomes.get(agent_id)
                    if not raw_outcome:
                        # Should theoretically not happen if logic is sound
                        continue
                        
                    # Combine parser reward (usually 0.0) + Env reward
                    total_reward = raw_outcome.reward + parse_result.reward
                    
                    outcome = StepOutcome(
                        obs=raw_outcome.obs,
                        reward=total_reward,
                        truncated=raw_outcome.truncated,
                        terminated=raw_outcome.terminated,
                        info={**client_info, **raw_outcome.info}
                    )

                # Update flags for global loop break
                if outcome.terminated: any_term = True
                if outcome.truncated: any_trunc = True

                # 1. Log the CLEAN single-agent step to the collector
                step = Step(
                    index=t,
                    prev_obs=ctx["prev_obs"],
                    action=ctx["raw_action"],
                    next_obs=outcome.obs if not (outcome.terminated or outcome.truncated) else None,
                    reward=outcome.reward,
                    truncated=outcome.truncated,
                    terminated=outcome.terminated,
                    info=outcome.info
                )
                collector.add(agent_id, step)

                # 2. Update Agent Context for the next turn
                # (Only if the episode isn't over for them)
                if not (outcome.terminated or outcome.truncated):
                    self.agent_map[agent_id].on_after_step(outcome.obs, outcome.info)
                    current_obs[agent_id] = outcome.obs

            # --- F. Global Termination Check ---
            # If any agent terminated/truncated, we generally consider the episode 
            # done for the group in this simple protocol.
            if any_term or any_trunc:
                break
                        
        return collector.extract_rollouts()
