from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from ludic.env import Env
from ludic.types import Rollout, SamplingArgs

class InteractionProtocol(ABC):
    """
    Abstract base class for all interaction protocols.
    
    A protocol defines the "rules of the game" for how Agent(s) and an
    EnvKernel interact. The protocol is initialized with the agent(s)
    it will manage.
    """
    
    @abstractmethod
    async def run(
        self,
        *,
        env: Env,
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
    ) -> Rollout:
        """
        Executes one full episode according to the protocol's rules
        and returns the complete Rollout.

        Args:
            env: The environment instance to run against.
            max_steps: Maximum number of steps for the episode.
            seed: Optional seed for env.reset().
            sampling_args: Optional sampling config for this run.
            timeout_s: Optional timeout for each agent.act() call.
        """
        ...