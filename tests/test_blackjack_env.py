import random

from environments.blackjack import BlackjackEnv


def test_blackjack_seed_is_deterministic() -> None:
    env1 = BlackjackEnv()
    env2 = BlackjackEnv()

    obs1, info1 = env1.reset(seed=123)["agent_0"]
    obs2, info2 = env2.reset(seed=123)["agent_0"]

    assert obs1 == obs2
    assert info1["player_hand"] == info2["player_hand"]
    assert info1["dealer_upcard"] == info2["dealer_upcard"]


def test_blackjack_invalid_action_terminates_not_truncates() -> None:
    env = BlackjackEnv()
    env.reset(seed=0)

    outcome = env.step({"agent_0": "banana"})["agent_0"]

    assert outcome.reward == -1.0
    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.info.get("invalid_action") is True


def test_blackjack_reset_does_not_touch_global_random_state() -> None:
    env = BlackjackEnv()

    state = random.getstate()
    try:
        random.seed(123)
        before = random.random()

        env.reset(seed=0)

        after = random.random()

        random.seed(123)
        expected_before = random.random()
        expected_after = random.random()

        assert before == expected_before
        assert after == expected_after
    finally:
        random.setstate(state)

