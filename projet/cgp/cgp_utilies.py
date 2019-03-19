import json

from cgp.cgp_model import Genome, GenomeConfig, Population


def deserialize_population(serialized_pop: [], genome_config: GenomeConfig, real_adn=False) -> Population:
    new_genomes_list = []
    scores = []
    for e in serialized_pop:
        if real_adn:
            new_genomes_list.append(Genome.from_list_of_dict(json.loads(decode(e))[1], genome_config))
            scores.append(json.loads(decode(e))[0])
        else:
            new_genomes_list.append(Genome.from_list_of_dict(e[1], genome_config))
            scores.append(e[0])
    pop = Population(genome_config, len(new_genomes_list))
    pop.list_genomes = new_genomes_list
    pop.list_scores = scores
    return pop

def merge_populations(pop: Population, pops: [Population]):
    lg = pop.list_genomes
    ls = pop.list_scores
    for p in pops:
        lg += p.list_genomes
        ls += p.list_scores
    # remove duplicate
    filtered_g = []
    filtered_s = []
    for i in range(len(lg)):
        gen1 = lg[i]
        can_add = True
        for j in range(len(filtered_g)):
            gen2 = filtered_g[j]
            if gen1 == gen2:
                can_add = False
        if can_add:
            filtered_g.append(lg[i])
            filtered_s.append(ls[i])
    pop.list_genomes = filtered_g
    pop.list_scores = filtered_s

def load_population(genome_config: GenomeConfig, save_name: str, save_dir="./saves/", real_adn=False):
    with open("{}{}.json".format(save_dir, save_name), "r") as infile:
        return deserialize_population(json.load(infile), genome_config, real_adn=real_adn)


def encode(s: str):
    res = str()
    switch = {
        "0": "AA",
        "1": "AC",
        "2": "AG",
        "3": "AT",
        "4": "CA",
        "5": "CC",
        "6": "CG",
        "7": "CT",
        "8": "GA",
        "9": "GC",
        "a": "GG",
        "b": "GT",
        "c": "TA",
        "d": "TC",
        "e": "TG",
        "f": "TT",
    }
    for i in s.encode("utf-8").hex():
        res += switch[i]
    return res


def decode(s: str):
    res = str()
    switch = {
        "AA": "0",
        "AC": "1",
        "AG": "2",
        "AT": "3",
        "CA": "4",
        "CC": "5",
        "CG": "6",
        "CT": "7",
        "GA": "8",
        "GC": "9",
        "GG": "a",
        "GT": "b",
        "TA": "c",
        "TC": "d",
        "TG": "e",
        "TT": "f",
    }
    for i in [s[i:i + 2] for i in range(0, len(s), 2)]:
        res += switch[i]
    return bytearray.fromhex(res).decode()
