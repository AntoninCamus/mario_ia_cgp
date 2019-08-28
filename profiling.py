#!/usr/bin/env python3

import cProfile

from cgp import cgp_utilies
from mario_emu.emulator import Emulator
from mario_emu.picture_processor import PictureProcessor

def _prof_run():
    TEST_NB = 10
    POPULATION_SIZE = 25
    for i in range(TEST_NB):
        print("[PROFILING] Step {}/{}".format(i + 1, TEST_NB))
        cfg = cgp_utilies.GenomeConfig(
            PictureProcessor.get_dim(),
            Emulator.get_action_space_size(),
            512
        )
        pop = cgp_utilies.Population(cfg, POPULATION_SIZE)
        Emulator.eval_population(pop)
        pop.next_gen()

if __name__ == '__main__':
    cProfile.run("_prof_run()")


