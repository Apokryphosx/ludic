from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Info, Observation, StepOutcome


class BlackjackEnv(SingleAgentEnv):
    """
    Simple single-player Blackjack environment.

    Rules implemented here:
      * 52-card shoe (optionally multiple decks)
      * Player actions: HIT or STAND (case-insensitive)
      * Dealer stands on soft 17 (configurable)
      * Rewards: +1 win, -1 loss, 0 push, -1 for invalid move
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are playing Blackjack against a dealer.\n"
        "On each turn, choose whether to take another card or hold.\n"
        "Reply with exactly one XML tag: <move>HIT</move> or <move>STAND</move>."
    )

    RANKS: Sequence[str] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
    SUITS: Sequence[str] = ("S", "H", "D", "C")  # Spades, Hearts, Diamonds, Clubs
    HIT_ALIASES = {"hit", "h"}
    STAND_ALIASES = {"stand", "s", "stay"}

    def __init__(
        self,
        *,
        decks: int = 1,
        dealer_hits_soft_17: bool = False,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        super().__init__()
        if decks <= 0:
            raise ValueError("`decks` must be >= 1")
        self._decks = decks
        self._dealer_hits_soft_17 = dealer_hits_soft_17
        self._system_prompt = system_prompt

        self._rng = random.Random()
        self._deck: List[str] = []
        self._player_cards: List[str] = []
        self._dealer_cards: List[str] = []
        self._done: bool = False
        self._last_obs: Observation = ""

    # ------------------------------------------------------------------
    # SingleAgentEnv API
    # ------------------------------------------------------------------

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return self._system_prompt

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        if seed is not None:
            self._rng.seed(seed)

        self._deck = self._build_shoe()
        self._player_cards = [self._draw_card(), self._draw_card()]
        self._dealer_cards = [self._draw_card(), self._draw_card()]
        self._done = False

        obs = self._render_obs()
        info: Info = {
            "player_hand": list(self._player_cards),
            "dealer_upcard": self._dealer_cards[0],
        }
        return obs, info

    def env_step(self, action: str) -> StepOutcome:
        if self._done:
            raise RuntimeError("BlackjackEnv.step() called after episode finished. Call reset().")

        normalized = action.strip().lower()
        info: Info = {"raw_action": action}

        if normalized in self.HIT_ALIASES:
            return self._handle_hit(info)
        if normalized in self.STAND_ALIASES:
            return self._handle_stand(info)

        # Invalid action immediately ends the round.
        self._done = True
        obs = "Invalid action. Please respond with HIT or STAND."
        self._last_obs = obs
        info.update({"invalid_action": True})
        return StepOutcome(
            obs=obs,
            reward=-1.0,
            truncated=False,
            terminated=True,
            info=info,
        )

    def env_current_obs(self) -> Observation:
        return self._last_obs

    # ------------------------------------------------------------------
    # Gameplay helpers
    # ------------------------------------------------------------------

    def _handle_hit(self, info: Info) -> StepOutcome:
        card = self._draw_card()
        self._player_cards.append(card)
        player_total, _ = self._hand_value(self._player_cards)
        info.update(
            {
                "action": "hit",
                "drawn_card": card,
                "player_hand": list(self._player_cards),
                "player_total": player_total,
            }
        )

        if player_total > 21:
            self._done = True
            obs = self._render_obs(reveal_dealer=True, final_note="You busted.")
            self._last_obs = obs
            info.update({"result": "player_bust"})
            return StepOutcome(
                obs=obs,
                reward=-1.0,
                truncated=False,
                terminated=True,
                info=info,
            )

        obs = self._render_obs()
        self._last_obs = obs
        return StepOutcome(
            obs=obs,
            reward=0.0,
            truncated=False,
            terminated=False,
            info=info,
        )

    def _handle_stand(self, info: Info) -> StepOutcome:
        self._done = True
        dealer_draws: List[str] = []
        while self._dealer_should_hit():
            card = self._draw_card()
            self._dealer_cards.append(card)
            dealer_draws.append(card)

        player_total, _ = self._hand_value(self._player_cards)
        dealer_total, _ = self._hand_value(self._dealer_cards)

        if dealer_total > 21:
            result = "dealer_bust"
            reward = 1.0
            final_note = "Dealer busted. You win!"
        elif dealer_total < player_total:
            result = "player_win"
            reward = 1.0
            final_note = "You beat the dealer!"
        elif dealer_total > player_total:
            result = "dealer_win"
            reward = -1.0
            final_note = "Dealer wins."
        else:
            result = "push"
            reward = 0.0
            final_note = "Push."

        info.update(
            {
                "action": "stand",
                "result": result,
                "player_hand": list(self._player_cards),
                "dealer_hand": list(self._dealer_cards),
                "player_total": player_total,
                "dealer_total": dealer_total,
                "dealer_draws": dealer_draws,
            }
        )

        obs = self._render_obs(reveal_dealer=True, final_note=final_note)
        self._last_obs = obs
        return StepOutcome(
            obs=obs,
            reward=reward,
            truncated=False,
            terminated=True,
            info=info,
        )

    def _build_shoe(self) -> List[str]:
        deck: List[str] = []
        for _ in range(self._decks):
            for suit in self.SUITS:
                for rank in self.RANKS:
                    deck.append(f"{rank}{suit}")
        self._rng.shuffle(deck)
        return deck

    def _draw_card(self) -> str:
        if not self._deck:
            self._deck = self._build_shoe()
        return self._deck.pop()

    def _hand_value(self, cards: Sequence[str]) -> Tuple[int, bool]:
        total = 0
        soft_aces = 0
        for card in cards:
            rank = card[:-1]
            if rank == "A":
                total += 11
                soft_aces += 1
            elif rank in {"K", "Q", "J", "10"}:
                total += 10
            else:
                total += int(rank)

        # Convert soft aces to hard aces if we are busting.
        used_soft_aces = soft_aces
        while total > 21 and used_soft_aces > 0:
            total -= 10
            used_soft_aces -= 1

        is_soft = used_soft_aces > 0
        return total, is_soft

    def _dealer_should_hit(self) -> bool:
        total, is_soft = self._hand_value(self._dealer_cards)
        if total < 17:
            return True
        if total > 21:
            return False
        if total == 17 and is_soft and self._dealer_hits_soft_17:
            return True
        return False

    def _format_hand(self, cards: Sequence[str]) -> str:
        return ", ".join(cards)

    def _render_obs(
        self,
        *,
        reveal_dealer: bool = False,
        final_note: Optional[str] = None,
    ) -> Observation:
        player_total, player_soft = self._hand_value(self._player_cards)
        player_desc = f"Your hand ({player_total}{' soft' if player_soft else ''}): {self._format_hand(self._player_cards)}"

        if reveal_dealer or self._done:
            dealer_total, dealer_soft = self._hand_value(self._dealer_cards)
            dealer_desc = (
                f"Dealer hand ({dealer_total}{' soft' if dealer_soft else ''}): "
                f"{self._format_hand(self._dealer_cards)}"
            )
        else:
            dealer_desc = (
                f"Dealer shows {self._dealer_cards[0]} and a facedown card."
            )

        lines = [
            "You are playing Blackjack versus a single dealer.",
            player_desc,
            dealer_desc,
        ]

        if not (reveal_dealer or self._done):
            lines.append("Available actions: HIT to draw another card, STAND to hold.")

        if final_note is not None:
            lines.append(f"Round result: {final_note}")

        obs = "\n".join(lines)
        self._last_obs = obs
        return obs
