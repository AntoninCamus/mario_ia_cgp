import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

from cgp.cgp_model import Population, Genome
from mario_emu.picture_processor import PictureProcessor


class Emulator:
    def __init__(self):
        self.env = gym_super_mario_bros.make("SuperMarioBros-2-1-v2")
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)

    def _make_it_play(self, g: Genome, render: bool) -> float:
        observation = self.env.reset()
        total_reward = 0.0
        stuck_score = 0
        for i in range(1, 10000):
            if render:
                self.env.render()
            ob_flat = PictureProcessor.process(observation)
            decision = g.evaluate(ob_flat)
            action = decision.argmax()
            # action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            stuck_score += reward
            if total_reward < 0:
                break
            if i % 100 == 0:
                if stuck_score <= 0:
                    break
                stuck_score = 0
            if done:
                break
        return total_reward

    @staticmethod
    def eval_population(p: Population, render=False, debug=False):
        e = Emulator()
        for i in range(len(p.list_genomes)):
            if p.list_scores[i] is None:
                p.list_scores[i] = e._make_it_play(p.list_genomes[i], render)
            if debug:
                print("Genome {} got reward {}".format(p.list_genomes[i], p.list_scores[i]))
        # Get 5 (or keep) best genome by sorting by mark the list, slice it and reconstruct it