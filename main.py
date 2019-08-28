#!/usr/bin/env python3

import argparse
from os import system

import cgp.cgp_utilies
from controller import get_top, post_top
from mario_emu.emulator import Emulator

# Hyper parameters
from mario_emu.picture_processor import PictureProcessor

POPULATION_SIZE = 100
GENOME_SIZE = 1024
NB_BREED = 5  # Number of children generated from know-good parents
NB_MUTATIONS = 20  # Number of mutation per genome applied from a parent to a children

# Saving parameters
SAVE_EVERY = 5  # Number of states between saves
KEEP = 10  # Number of good children to pass to the next gen


def learn(mpi_comm=None, debug=False, render=False, save_name=None, host_name=None):
    if mpi_comm:
        rank = mpi_comm.Get_rank()
    else:
        rank = 0

    cfg = cgp.cgp_utilies.GenomeConfig(PictureProcessor.get_dim(), Emulator.get_action_space_size(), GENOME_SIZE)

    if save_name is None:
        pop = cgp.cgp_utilies.Population(cfg, POPULATION_SIZE, KEEP)
    else:
        pop = cgp.cgp_utilies.load_population(cfg, save_name)
        pop.size = POPULATION_SIZE
        pop.keep = KEEP
        pop.next_gen()

    i = 0
    while True:
        if mpi_comm and debug:
            print("Testing generation {} on worker {}...".format(i, rank))
        else:
            if rank == 0:
                print("Testing generation {}...".format(i))

        Emulator.eval_population(pop, render=render, debug=debug)
        pop.keep_bests()

        print("Best scores were {}...".format(pop.list_scores[:5]))
        # All Gathering :
        if i % SAVE_EVERY == 0:
            if host_name:
                pop = cgp.cgp_utilies.deserialize_population(
                    post_top(host_name, pop.serialize()), pop.genome_config
                )
                if debug:
                    print("Saving on server {}, new state is {}...".format(host_name, pop))
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
        i += 1


def visualize(save_name):
    cfg = cgp.cgp_utilies.GenomeConfig(PictureProcessor.get_dim(), Emulator.get_action_space_size(), GENOME_SIZE)
    pop = cgp.cgp_utilies.load_population(cfg, save_name)
    for (i, gen) in enumerate(pop.list_genomes):
        gen.compute_used_node()
        print("Genome", i, ":", sum(gen.used_node))
        infile = "viz/{}_{}".format(save_name, i)
        with open(infile + ".gv", "w") as f:
            f.write("digraph gen{} {{\n".format(i))
            for j in range(gen.conf.inp):
                f.write('\t{} [label="{}"];\n'.format(j, "input " + str(j)))
                f.write("\t{} -> {};\n".format("Input", j))
            for j in range(gen.conf.inp, gen.conf.inp + gen.conf.node):
                if gen.used_node[j]:
                    f.write(
                        '\t{} [label="{}"];\n'.format(
                            j, "{}: {}".format(j, gen.genotype[j - gen.conf.inp].fn_name())
                        )
                    )
                    for p in gen.genotype[j - gen.conf.inp].preds():
                        f.write("\t{} -> {};\n".format(p, j))
            for j in range(
                gen.conf.inp + gen.conf.node, gen.conf.inp + gen.conf.node + gen.conf.out
            ):
                f.write(
                    '\t{} [label="{}"];\n'.format(
                        j,
                        "output {}: {}".format(
                            j - gen.conf.node - gen.conf.inp,
                            gen.genotype[j - gen.conf.inp].fn_name(),
                        ),
                    )
                )
                f.write("\t{} -> {};\n".format(j, "Decision"))
                for p in gen.genotype[j - gen.conf.inp].preds():
                    f.write("\t{} -> {};\n".format(p, j))

            f.write("}")
        system("dot -Tpng {}.gv -o {}.png".format(infile, infile))


def main():
    # PARSER CONFIGURATION
    parser = argparse.ArgumentParser(description="Cartesian Genetic Program playing Mario Bros :3")
    parser.add_argument("-l", "--load", help="Load a save file.")
    parser.add_argument("-d", "--debug", help="Display debugging messages.", action="store_true")
    parser.add_argument(
        "-r", "--render", help="Display screen during playing.", action="store_true"
    )
    parser.add_argument(
        "-c", "--collector", help="Use a remote server to share results between workers."
    )
    parser.add_argument(
        "-m", "--mpi", help="Start the program in distributed with MPI.", action="store_true"
    )
    parser.add_argument(
        "-v",
        "--visualize",
        help="Generate graphs for given save name, should NOT be used with other options.",
    )

    # PARSING
    args = parser.parse_args()

    if args.visualize:
        visualize(args.visualize)
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


if __name__ == "__main__":
    main()
