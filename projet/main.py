import argparse
import cProfile
import sys

import cgp.cgp_utilies
import emu.picture_processing
from distributed.controller import get_top, post_top
from emu.emu_env import EmuEnv

# Hyper parameters
NB_GENS = 10000
POP_SIZE = 200
GENOME_SIZE = 512
BREED = 2
MUTA = 20

# Saving parameters
SAVE_EVERY = 1
KEEP = 5

# Program parameters
debug = False
render = False
image_processor = emu.picture_processing.PictureReducer()


def learn(mpi_comm=None, save_name=None, host_name=None):
    if mpi_comm:
        rank = mpi_comm.Get_rank()
    else:
        rank = 0

    cfg = cgp.cgp_utilies.GenomeConfig(image_processor.get_dim(), 7, GENOME_SIZE)

    if save_name is None:
        pop = cgp.cgp_utilies.Population(cfg, POP_SIZE, KEEP)
    else:
        pop = cgp.cgp_utilies.load_population(cfg, save_name)
        pop.size = POP_SIZE
        pop.keep = KEEP
        pop.next_gen()

    for i in range(1, NB_GENS):
        if mpi_comm and debug:
            print("Testing generation {} on worker {}...".format(i, rank))
        else:
            if rank == 0:
                print("Testing generation {}...".format(i))

        EmuEnv.make_them_play(pop, image_processor, render=render, debug=debug)
        pop.keep_bests()

        print("Best scores were {}...".format(pop.list_scores[:5]))
        # All Gathering :
        if i % SAVE_EVERY == 0:
            if host_name:
                pop = cgp.cgp_utilies.deserialize_population(
                    post_top(host_name, pop.serialize()),
                    pop.genome_config
                )
                if debug:
                    print("Saving on server {}, new state is {}...".format(host_name, pop))
            elif mpi_comm:
                shared = [cgp.cgp_utilies.deserialize_population(p, pop.genome_config) for p in
                          mpi_comm.allgather(pop.serialize())]
                cgp.cgp_utilies.merge_populations(pop, shared)

            if rank == 0:
                save_name = "save_" + str(i)
                if debug:
                    print("Saving in {}...".format(save_name))
                pop.save(save_name)

        pop.next_gen()


def profile_run():
    for i in range(2):
        cfg = cgp.cgp_utilies.GenomeConfig(image_processor.get_dim(), 7, 1000)
        pop = cgp.cgp_utilies.Population(cfg, 10)
        EmuEnv.make_them_play(pop, image_processor, render=render)
        pop.next_gen()


def profile():
    cProfile.run('profile_run()')


def main():
    parser = argparse.ArgumentParser(description='Cartesian Genetic Program playing Mario Bros :3')
    parser.add_argument("-l", "--load", help="Charge une sauvegarde")
    parser.add_argument("-c", "--collector", help="Indique l'ip du serveur de partage")
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

    if args.collector:
        print("Using host : {}".format(args.collector))
        collector_ip = args.collector
        print("Actual state of server is : {}".format(get_top(collector_ip)))
    else:
        collector_ip = None

    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    else:
        comm = None

    if args.profile:
        print("Starting profiling")
        profile()
        sys.exit()
    else:
        if not comm:
            print("Starting learning")
        elif comm.Get_rank() == 0:
            print("Starting learning with MPI")

        learn(comm, save_name=save_name, host_name=collector_ip)
        sys.exit()


if __name__ == "__main__":
    main()
