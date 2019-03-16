import argparse
import cProfile
import sys

from CgpModel import Population, generate_population_from, GenomeConfig
from Emu import EmuEnv
from PictureProcessing import *

NB_GENS = 10000
POP_SIZE = 50
SAVE_EVERY = 5

debug = False
render = False


def learn():
    image_processor = PictureReducer()
    cfg = GenomeConfig(image_processor.get_dim(), 7, row=64, col=64)  # TODO TO TUNE
    pop = Population(cfg, size=POP_SIZE)
    for i in range(NB_GENS):
        if debug:
            print("Testing generation {} : {}...".format(i, pop))
        bests = EmuEnv.make_them_play(pop, image_processor, render=render, debug=debug)
        if debug:
            print("Best of them were {}...".format(bests))
        pop = generate_population_from(bests, POP_SIZE, cfg)
        if i % SAVE_EVERY == 0:
            save_name = "save_" + str(i)
            if debug:
                print("Saving in {}...".format(save_name))
            pop.save(save_name)


def profile_run():
    for i in range(2):
        image_processor = PictureReducer()
        cfg = GenomeConfig(image_processor.get_dim(), 7, row=64, col=64)
        bests = EmuEnv.make_them_play(Population(cfg, size=10), image_processor, render=render)
        generate_population_from(bests, 5, cfg)


def profile():
    cProfile.run('profile_run()')


def main():
    parser = argparse.ArgumentParser(description='Cartesian Genetical Program playing Mario Bros :3')
    parser.add_argument("-d", "--debug", help="Affiche les textes de débogage", action="store_true")
    parser.add_argument("-r", "--render", help="Affiche les parties en cours", action="store_true")
    parser.add_argument("-p", "--profile", help="Profile un run pour trouver les bottlenecks de l'algorithme",
                        action="store_true")
    args = parser.parse_args()
    if args.debug:
        print("Debug mode activated")
        global debug
        debug = True
    if args.render:
        print("Rendering activated")
        global render
        render = True
    if args.profile:
        print("Starting profiling")
        profile()
        sys.exit()
    else:
        print("Starting learning")
        learn()
        sys.exit()


if __name__ == "__main__":
    main()
