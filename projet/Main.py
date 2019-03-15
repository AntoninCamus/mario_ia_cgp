import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

from CgpModel import Population, generate_population_from, GenomeConfig
from Emu import EmuEnv

NB_GENS = 10000
POP_SIZE = 50
SAVE_EVERY = 5


class Game:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-2-1-v3')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)


def main():
    cfg = GenomeConfig(23, 32)  # TO TUNE
    pop = Population(cfg, size=POP_SIZE)
    for i in range(NB_GENS):
        bests = EmuEnv.make_them_play(pop)
        pop = generate_population_from(bests)
        if i % SAVE_EVERY == 0:
            pop.save("save_" + str(i))

if __name__ == "__main__":
    main()
