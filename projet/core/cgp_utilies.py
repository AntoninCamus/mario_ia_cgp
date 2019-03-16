import json
from random import randint

from core.cgp_model import Genome, GenomeConfig


def serialize_genomes_list(genomes_list: [Genome], real_adn=False) -> []:
    serialized_list = []
    for g in genomes_list:
        if real_adn:
            serialized_list.append(encode(json.dumps(g.to_list_of_dict())))
        else:
            serialized_list.append(g.to_list_of_dict())
    return serialized_list


def save_genomes_list(list_genomes: [Genome], save_name: str, save_dir="./saves/", real_adn=False):
    with open("{}{}.json".format(save_dir, save_name), "w+") as outfile:
        json.dump(serialize_genomes_list(list_genomes, real_adn=real_adn), outfile)


def deserialize_genomes_list(serialized_list: [], genome_config: GenomeConfig, real_adn=False) -> [Genome]:
    new_genomes_list = []
    for e in serialized_list:
        if real_adn:
            new_genomes_list.append(Genome.from_list_of_dict(json.loads(decode(e)), genome_config))
        else:
            new_genomes_list.append(Genome.from_list_of_dict(e, genome_config))
    return new_genomes_list


def load_genomes_list(genome_config: GenomeConfig, save_name: str, save_dir="./saves/", real_adn=False):
    with open("{}{}.json".format(save_dir, save_name), "r") as infile:
        return deserialize_genomes_list(json.load(infile), genome_config, real_adn=real_adn)


def generate_genomes_from(bests: [Genome], pop_size: int, muta_count=10) -> [Genome]:
    # give birth bests.size - pop_size from bests winner by mutation
    return bests + [bests[randint(0, len(bests) - 1)].clone().mutate(muta_count) for _ in
                    range(pop_size - len(bests))]


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
