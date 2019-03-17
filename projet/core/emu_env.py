import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

from core.cgp_model import Genome


class EmuEnv:
    def __init__(self, processor):
        self.env = gym_super_mario_bros.make('SuperMarioBros-2-1-v3')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
        self.processor = processor

    def _make_it_play(self, g: Genome, render: bool) -> int:
        observation = self.env.reset()
        total_reward = 0.0
        for i in range(500):
            if render:
                self.env.render()
            ob_flat = self.processor.process(observation).tolist()
            decision = g.evaluate(ob_flat)
            action = decision.argmax()  # TODO Optimize function by using np.array and optimize picture
            # action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                total_reward /= i
                break
            pass
        return total_reward

    @staticmethod
    def make_them_play(list_genomes: [Genome], processor, keep=5, render=False, debug=False):
        e = EmuEnv(processor)
        results: [int] = []
        for i in range(len(list_genomes)):
            res = e._make_it_play(list_genomes[i], render), list_genomes[i]
            if debug:
                print("Genome {} got reward {}".format(res[1], res[0]))
            results.insert(i, res)
        # Get 5 (or keep) best genome by sorting by mark the list, slice it and reconstruct it
        return [t[1] for t in sorted(results, key=lambda x: x[0])[0:keep]]
