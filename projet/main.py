#!/usr/bin/env python3

import argparse
import cProfile
import sys

import cgp.cgp_utilies
import emu.picture_processing
from distributed.controller import get_top, post_top
from emu.emu_env import EmuEnv

# Hyper parameters
NB_GENS = 10000
POP_SIZE = 120
GENOME_SIZE = 1024
BREED = 3
MUTA = 20

# Saving parameters
SAVE_EVERY = 5
KEEP = 5

# Program parameters
image_processor = emu.picture_processing.PictureReducer()


def learn(mpi_comm=None, debug=False, render=False, save_name=None, host_name=None):
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
                    post_top(host_name, pop.serialize()), pop.genome_config
                )
                if debug:
                    print(
                        "Saving on server {}, new state is {}...".format(host_name, pop)
                    )
            elif mpi_comm:
                shared = [
                    cgp.cgp_utilies.deserialize_population(p, pop.genome_config)
                    for p in mpi_comm.allgather(pop.serialize())
                ]
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
        EmuEnv.make_them_play(pop, image_processor)
        pop.next_gen()


def profile():
    cProfile.run("profile_run()")


def main():
    # PARSER CONFIGURATION
    parser = argparse.ArgumentParser(
        description="Cartesian Genetic Program playing Mario Bros :3"
    )
    parser.add_argument("-l", "--load", help="Load a save file.")
    parser.add_argument("-c", "--collector", help="Use a remote server to share results between workers.")
    parser.add_argument(
        "-m",
        "--mpi",
        help="Start the program in distributed with MPI",
        action="store_true",
    )
    parser.add_argument(
        "-d", "--debug", help="Display debugging messages.", action="store_true"
    )
    parser.add_argument(
        "-r", "--render", help="Display screen during playing.", action="store_true"
    )
    parser.add_argument(
        "-p",
        "--profile",
        help="Start a profiling of performances, should NOT be used with other options.",
        action="store_true",
    )

    # PARSING
    args = parser.parse_args()

    if args.profile:
        if args.debug or args.load or args.mpi or args.collector or args.render:
            print(
                "Debug, load, MPI, collector, or render options are not compatible with profiling, aborting."
            )
            exit(1)
        print("Starting profiling")
        profile()
        sys.exit()
    else:

        # Configuring variables
        if args.mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        else:
            comm = None

        if not comm:
            print("Starting learning")
        elif comm.Get_rank() == 0:
            print("Starting learning with MPI")

        # Printing only once if you use mpi (avoid spamming stdout)
        if not comm or comm.Get_rank() == 0:
            if args.debug:
                print("Debug mode activated")
            if args.render:
                print("Rendering activated")
            if args.collector:
                print("Using host : {}".format(args.collector))
                print("Actual state of server is : {}".format(get_top(args.collector)))
            if args.load:
                print("Loading save {}".format(args.load))

        learn(
            comm,
            debug=args.debug,
            render=args.render,
            save_name=args.load,
            host_name=args.collector,
        )
        sys.exit()


if __name__ == "__main__":
    main()
