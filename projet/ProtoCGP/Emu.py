import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

from ProtoCGP.CgpModel import Genome, Population
from ProtoCGP.PictureProcessing import PictureFlattener


class EmuEnv:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-2-1-v3')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)

    def _make_it_play(self, g: Genome) -> int:
        flattener = PictureFlattener()
        observation = self.env.reset()
        total_reward = 0.0
        for i in range(5000):
            self.env.render()
            action = self.env.action_space[g.evaluate(flattener.process(observation))]
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                total_reward /= i
                break
            pass
        return total_reward

    @staticmethod
    def _make_them_play(p: Population, keep=5):
        # Fonction a paralleliser avec cython si on veut opti le tout
        e = EmuEnv()
        results: [int] = []
        for i in range(len(p.list_genomes)):
            results[i] = (e._make_it_play(p.list_genomes[i]), p.list_genomes[i])
            # Paralleliser à l'aide de cython ou d'autre chose ?
            # http://nealhughes.net/parallelcomp2/
        # Get 5 (or keep) best genome by sorting by mark the list, slice it and reconstruct it
        return [t[1] for t in results.sort(key=lambda t: t[0])[:(keep + 1)]]
