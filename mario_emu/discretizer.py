import gym
import numpy as np
import retro


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        try:
            return self._decode_discrete_action[act].copy()
        except IndexError:
            raise IndexError("Action {} is out of range, there is only {} actions.".format(act, self.action_space.n))

class SuperMarioDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                ["LEFT"],
                ["RIGHT"],
                ["DOWN"],
                #["LEFT", "B"],
                #["RIGHT", "B"],
                #["B"],
                ["LEFT", "A"],
                ["RIGHT", "A"],
                ["A"],
            ],
        )