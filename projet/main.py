import argparse
import cProfile
import sys

from core.cgp_model import GenomeConfig, Genome
from core.emu_env import EmuEnv
from core.cgp_utilies import *
from core.picture_processing import *

# Hyper parameters
NB_GENS = 10000
POP_SIZE = 100
GENOME_SIZE = 512

# Saving parameters
SAVE_EVERY = 10
KEEP = 5

# Program parameters
debug = False
render = False
image_processor = PictureReducer()


def learn(save_name=None):
    cfg = GenomeConfig(image_processor.get_dim(), 7, GENOME_SIZE)

    if save_name is None:
        pop = [Genome(cfg) for _ in range(POP_SIZE)]
    else:
        loaded = load_genomes_list(cfg, save_name)
        pop = generate_genomes_from(loaded, POP_SIZE)

    for i in range(1, NB_GENS):
        if debug:
            print("Testing generation {} : {}...".format(i, pop))
        else:
            print("Testing generation {} ...".format(i))
        bests = EmuEnv.make_them_play(pop, image_processor, keep=KEEP, render=render, debug=debug)
        if debug:
            print("Best of them were {}...".format(bests))
        if i % SAVE_EVERY == 0:
            save_name = "save_" + str(i)
            if debug:
                print("Saving in {}...".format(save_name))
            save_genomes_list(bests, save_name)
        pop = generate_genomes_from(bests, POP_SIZE)


def learn_mpi(comm, save_name=None):
    rank = comm.Get_rank()

    cfg = GenomeConfig(image_processor.get_dim(), 7, GENOME_SIZE)

    if save_name is None:
        pop = [Genome(cfg) for _ in range(POP_SIZE)]
    else:
        loaded = load_genomes_list(cfg, save_name)
        pop = generate_genomes_from(loaded, POP_SIZE)

    for i in range(1, NB_GENS):
        if rank == 0:
            if debug:
                print("Testing generation {} : {}...".format(i, pop))
            else:
                print("Testing generation {} ...".format(i))
        if debug:
            print("Testing generation {} on worker {} : {}...".format(i, rank, pop))
        bests = EmuEnv.make_them_play(pop, image_processor, keep=KEEP, render=render, debug=debug)
        if debug:
            print("Best of them on worker {} were {}...".format(rank, bests))

        # All Gathering :
        shared = comm.allgather(serialize_genomes_list(bests))
        merged = [genome for genome_list in shared for genome in genome_list]
        bests = deserialize_genomes_list(merged, cfg)

        if rank == 0 and i % SAVE_EVERY == 0:
            save_name = "save_" + str(i)
            if debug:
                print("Saving in {}...".format(save_name))
            save_genomes_list(bests, save_name)
        pop = generate_genomes_from(bests, POP_SIZE)


def profile_run():
    for i in range(2):
        image_processor = PictureReducer()
        cfg = GenomeConfig(image_processor.get_dim(), 7, 1000)
        bests = EmuEnv.make_them_play([Genome(cfg) for _ in range(10)], image_processor, render=render)
        generate_genomes_from(bests, 10, cfg)


def profile():
    cProfile.run('profile_run()')


def main():
    parser = argparse.ArgumentParser(description='Cartesian Genetical Program playing Mario Bros :3')
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
