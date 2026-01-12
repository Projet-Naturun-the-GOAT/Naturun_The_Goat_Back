import numpy as np
import pytest

from src.ai_agent.q_learning import QLearningAgent


class DummyEnv:
    def __init__(self, level=1, has_key=False):
        self.level = level
        self.has_key = has_key


class DummyTrainEnv:
    def __init__(self, reward=1.0, level=1):
        self.reward = reward
        self.level = level
        self.reset_called = 0
        self.step_called = 0

    def reset(self):
        self.reset_called += 1
        return (0, 0)

    def step(self, _action):
        self.step_called += 1
        return (0, 1), self.reward, True, {}


class DummyTestEnv:
    def __init__(self, reward=5.0, level=1):
        self.reward = reward
        self.level = level
        self.reset_called = 0
        self.actions = []

    def reset(self):
        self.reset_called += 1
        return (0, 0)

    def step(self, action):
        self.actions.append(action)
        return (0, 1), self.reward, True, {}

    def render(self):
        return None


@pytest.fixture
def env_level_1():
    return DummyEnv(level=1)


@pytest.fixture
def env_level_2():
    return DummyEnv(level=2, has_key=False)


def test_get_full_state_level_1(env_level_1):
    agent = QLearningAgent(env=env_level_1)
    state = (2, 3)
    assert agent.get_full_state(state) == state


def test_get_full_state_level_2_includes_key(env_level_2):
    agent = QLearningAgent(env=env_level_2)
    state = (1, 1)
    assert agent.get_full_state(state) == (1, 1, False)
    env_level_2.has_key = True
    assert agent.get_full_state(state) == (1, 1, True)


def test_ensure_state_initializes_q_values(env_level_1):
    agent = QLearningAgent(env=env_level_1)
    state = (0, 0)
    agent.ensure_state(state)
    assert state in agent.q_table
    assert np.array_equal(agent.q_table[state], np.zeros(agent.action_size))


def test_choose_action_exploit_returns_argmax(env_level_1, monkeypatch):
    agent = QLearningAgent(env=env_level_1, epsilon=0.0)
    state = (0, 0)
    agent.q_table[state] = np.array([0.0, 2.0, 1.0, 1.5])
    monkeypatch.setattr("random.random", lambda: 1.0)
    assert agent.choose_action(state) == 1


def test_choose_action_explore_uses_random(env_level_1, monkeypatch):
    agent = QLearningAgent(env=env_level_1, epsilon=1.0)
    state = (0, 0)
    agent.q_table[state] = np.array([0.0, 2.0, 1.0, 1.5])
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.randint", lambda _low, _high: 2)
    assert agent.choose_action(state) == 2


def test_update_q_value(env_level_1):
    agent = QLearningAgent(env=env_level_1, alpha=0.5, gamma=0.9)
    state = (0, 0)
    next_state = (0, 1)
    agent.q_table[state] = np.zeros(agent.action_size)
    agent.q_table[next_state] = np.array([1.0, 2.0, 3.0, 4.0])

    agent.update(state, action=2, reward=1.0, next_state=next_state, done=False)

    expected = 0.5 * (1.0 + 0.9 * 4.0)
    assert agent.q_table[state][2] == pytest.approx(expected)


def test_update_q_value_when_done(env_level_1):
    agent = QLearningAgent(env=env_level_1, alpha=0.5, gamma=0.9)
    state = (0, 0)
    next_state = (0, 1)
    agent.q_table[state] = np.zeros(agent.action_size)
    agent.q_table[next_state] = np.array([1.0, 2.0, 3.0, 4.0])

    agent.update(state, action=1, reward=2.0, next_state=next_state, done=True)

    expected = 0.5 * 2.0
    assert agent.q_table[state][1] == pytest.approx(expected)


def test_decay_epsilon_respects_min(env_level_1):
    agent = QLearningAgent(
        env=env_level_1, epsilon=0.02, epsilon_min=0.01, epsilon_decay=0.5
    )
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.01)


def test_reset_exploration_sets_value(env_level_1):
    agent = QLearningAgent(env=env_level_1, epsilon=0.05)
    agent.reset_exploration(epsilon=0.3)
    assert agent.epsilon == pytest.approx(0.3)


def test_save_and_load_restores_state(env_level_1, tmp_path):
    agent = QLearningAgent(env=env_level_1, epsilon=0.2)
    agent.q_table[(0, 0)] = np.array([1.0, 2.0, 3.0, 4.0])
    agent.episodes_trained = 7

    model_path = tmp_path / "agent.npy"
    agent.save(str(model_path), best_reward=42.0)

    loaded_agent, saved_data = QLearningAgent.load(str(model_path), env=env_level_1)

    assert loaded_agent is not None
    assert loaded_agent.episodes_trained == 7
    assert loaded_agent.epsilon == pytest.approx(0.2)
    np.testing.assert_array_equal(
        loaded_agent.q_table[(0, 0)], np.array([1.0, 2.0, 3.0, 4.0])
    )
    assert loaded_agent.best_rewards[1] == pytest.approx(42.0)
    assert isinstance(saved_data, dict)


def test_load_missing_file_returns_none(env_level_1, tmp_path):
    missing_path = tmp_path / "missing.npy"
    loaded_agent, saved_data = QLearningAgent.load(str(missing_path), env=env_level_1)
    assert loaded_agent is None
    assert saved_data == {}


def test_train_updates_episode_and_best_reward(tmp_path):
    env = DummyTrainEnv(reward=1.0, level=1)
    agent = QLearningAgent(env=env, epsilon=1.0, epsilon_decay=0.5)
    model_path = tmp_path / "model.npy"

    agent.train(env, episodes=1, max_steps=1, model_filename=str(model_path))

    assert agent.episodes_trained == 1
    assert agent.epsilon == pytest.approx(0.5)
    assert agent.best_rewards[1] == pytest.approx(1.0)
    assert env.reset_called == 1
    assert env.step_called == 1


def test_test_method_restores_epsilon_and_returns_reward():
    env = DummyTestEnv(reward=5.0, level=1)
    agent = QLearningAgent(env=env, epsilon=0.4)
    agent.q_table[(0, 0)] = np.array([1.0, 0.0, 0.0, 0.0])

    total_reward, steps = agent.test(env, max_steps=5, render=False)

    assert total_reward == pytest.approx(5.0)
    assert steps == 1
    assert agent.epsilon == pytest.approx(0.4)
    assert env.actions == [0]
