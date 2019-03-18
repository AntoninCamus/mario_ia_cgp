import argparse
import cProfile
import sys

from core.cgp_model import GenomeConfig, Genome, Population
from core.emu_env import EmuEnv
from core.cgp_utilies import *
from core.picture_processing import *

# Hyper parameters
NB_GENS = 10000
POP_SIZE = 100
GENOME_SIZE = 512
BREED = 2
MUTA = 50

# Saving parameters
SAVE_EVERY = 5
KEEP = 5

# Program parameters
debug = False
render = False
image_processor = PictureReducer()


def learn(save_name=None):
    cfg = GenomeConfig(image_processor.get_dim(), 7, GENOME_SIZE)

    if save_name is None:
        pop = Population(cfg, POP_SIZE, KEEP, BREED, MUTA)
    else:
        pop = load_population(cfg, save_name)
        pop.size = POP_SIZE
        pop.keep = KEEP
        pop.next_gen()

    for i in range(1, NB_GENS):
        if debug:
            print("Testing generation {} : {}...".format(i, pop))
        else:
            print("Testing generation {} ...".format(i))
        EmuEnv.make_them_play(pop, image_processor, keep=KEEP, render=render, debug=debug)
        pop.keep_bests()
        if debug:
            print("Best scores were {}...".format(pop.list_scores[:5]))
        if i % SAVE_EVERY == 0:
            save_name = "save_" + str(i)
            if debug:
                print("Saving in {}...".format(save_name))
            pop.save(save_name)
        pop.next_gen()


def learn_mpi(comm, save_name=None):
    rank = comm.Get_rank()

    cfg = GenomeConfig(image_processor.get_dim(), 7, GENOME_SIZE)

    if save_name is None:
        pop = Population(cfg, POP_SIZE, KEEP)
    else:
        pop = load_population(cfg, save_name)
        pop.size = POP_SIZE
        pop.keep = KEEP
        pop.next_gen()

    for i in range(1, NB_GENS):
        if rank == 0:
            if debug:
                print("Testing generation {} : {}...".format(i, pop))
            else:
                print("Testing generation {} ...".format(i))
        if debug:
            print("Testing generation {} on worker {} : {}...".format(i, rank, pop))
        EmuEnv.make_them_play(pop, image_processor, keep=KEEP, render=render, debug=debug)
        pop.keep_bests()
        if debug:
            print("Best scores were {}...".format(pop.list_scores[:5]))
        
        # All Gathering :
        if i % SAVE_EVERY == 0:
            shared = [deserialize_population(p, pop.genome_config) for p in comm.allgather(pop.serialize())]
            merge_populations(pop, shared)
            if rank == 0:
                save_name = "save_" + str(i)
                if debug:
                    print("Saving in {}...".format(save_name))
                pop.save(save_name)
        pop.next_gen()


def profile_run():
    for i in range(2):
        cfg = GenomeConfig(image_processor.get_dim(), 7, 1000)
        pop = Population(cfg, 10)
        EmuEnv.make_them_play(pop, image_processor, render=render)
        pop.next_gen()


def profile():
    cProfile.run('profile_run()')


def main():
    parser = argparse.ArgumentParser(description='Cartesian Genetic Program playing Mario Bros :3')
    parser.add_argument("-l", "--load", help="Charge une sauvegarde")
    parser.add_argument("-m", "--mpi", help="Se lance en distribué MPI, incompatible avec le profiling",
                        action="store_true")
    parser.add_argument("-d", "--debug", help="Affiche les textes de débogage", action="store_true")
    parser.add_argument("-r", "--render", help="Affiche les parties en cours", action="store_true")
    parser.add_argument("-p", "--profile", help="Profile un run, incompatible avec le lancement via mpi",
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
    if args.load:
        print("Loading save {}".format(args.load))
        save_name = args.load
    else:
        save_name = None

    if args.profile:
        print("Starting profiling")
        profile()
        sys.exit()
    elif args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            print("Starting learning with MPI")
        learn_mpi(comm, save_name)
        sys.exit()
    else:
        print("Starting learning")
        learn(save_name)
        sys.exit()


if __name__ == "__main__":
    main()
