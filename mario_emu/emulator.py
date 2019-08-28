import retro

from cgp.cgp_model import Population, Genome
from mario_emu.discretizer import SuperMarioDiscretizer
from mario_emu.picture_processor import PictureProcessor

class Emulator:
    def __init__(self):
        self.env = retro.make('SuperMarioBros-Nes')
        self.disc = SuperMarioDiscretizer(self.env)

    def _evaluate_genome(self, g: Genome, render: bool = False, debug=False) -> float:
        observation = self.env.reset()
        total_reward = 0.0
        stuck_score = 0
        for i in range(1, 10000):
            if render:
                self.env.render()
            ob_flat = PictureProcessor.process(observation)
            action = self.disc.action(g.evaluate(ob_flat).argmax())
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
        if debug:
            print("Genome {} got reward {}".format(p.list_genomes[i], p.list_scores[i]))
        return total_reward

    @staticmethod
    def eval_population(p: Population, render=False, debug=False):
        e = Emulator()
        for i in range(len(p.list_genomes)):
            if p.list_scores[i] is None:
                p.list_scores[i] = e._evaluate_genome(p.list_genomes[i], render, debug)

    @staticmethod
    def get_action_space_size():
        return SuperMarioDiscretizer(retro.make('SuperMarioBros-Nes')).action_space.n
