from random import randint,choice
import json

import numpy as np

_default_func = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y if y != 0 else x
]


class GenomeConfig:
    def __init__(self, inp: int, out: int, node=1, func=None):
        if func is None:
            func = _default_func
        self.inp = inp
        self.out = out
        self.node = node
        self.func = func


class Neurone:
    def __init__(self, genome_config: GenomeConfig, node_id: int):
        self.node_id = node_id
        self.conf = genome_config
        self.pred1 = randint(0, genome_config.inp + node_id - 1)
        self.pred2 = randint(0, genome_config.inp + node_id - 1)
        self.func = randint(0, len(genome_config.func) - 1)

    def evaluate(self, data: [int]):
        val1 = data[self.pred1]
        val2 = data[self.pred2]
        return self.conf.func[self.func](val1, val2)

    def clone(self):
        res = Neurone(self.conf, self.node_id)
        res.pref1 = self.pred1
        res.pref2 = self.pred2
        res.func = self.func
        return res

    def mutate(self):
        rand = randint(0, 2)
        if rand == 0:
            self.pred1 = randint(0, self.conf.inp + self.node_id - 1)
        elif rand == 1:
            self.pred2 = randint(0, self.conf.inp + self.node_id - 1)
        else:
            self.func = randint(0, len(self.conf.func) - 1)

    def to_dict(self) -> dict:
        return {
            "type": "Neurone",
            "node_id": self.node_id,
            "pred1": self.pred1,
            "pred2": self.pred2,
            "func": self.func
        }

    def __eq__(self, other):
        return self.pred1 == other.pred1 and self.pred2 == other.pred2 and self.func == other.func

    @staticmethod
    def from_dict(d: dict, genome_config: GenomeConfig):
        res = Neurone(genome_config, d["node_id"])
        res.pred1 = d["pred1"]
        res.pred2 = d["pred2"]
        res.func = d["func"]
        return res


class OutputNeurone:
    def __init__(self, genome_config: GenomeConfig):
        self.conf = genome_config
        self.pred = randint(0, genome_config.inp + genome_config.node - 1)

    def evaluate(self, data: [int]):
        return data[self.pred]

    def clone(self):
        res = OutputNeurone(self.conf)
        res.pred = self.pred
        return res

    def mutate(self):
        self.pred = randint(0, self.conf.inp + self.conf.node - 1)

    def to_dict(self) -> dict:
        return {
            "type": "OutputNeurone",
            "pred": self.pred
        }

    def __eq__(self, other):
        return self.pred == other.pred

    @staticmethod
    def from_dict(d: dict, genome_config: GenomeConfig):
        res = OutputNeurone(genome_config)
        res.pred = d["pred"]
        return res


class Genome:
    def __init__(self, genome_config: GenomeConfig):
        self.conf = genome_config
        self.genotype = [Neurone(genome_config, i) for i in range(genome_config.node)]
        self.genotype += [OutputNeurone(genome_config) for _ in range(genome_config.out)]
        self.used_node = np.empty((), dtype=bool)
        self.compute_used_node()

    def compute_used_node(self) -> [bool]:
        self.used_node = np.zeros((self.conf.inp + self.conf.node + self.conf.out,), dtype=bool)
        for i in range(self.conf.node + self.conf.out - 1, -1, -1):
            if type(self.genotype[i]) is OutputNeurone:
                self.used_node[i + self.conf.inp] = True
            if self.used_node[i + self.conf.inp]:
                if type(self.genotype[i]) is OutputNeurone:
                    self.used_node[self.genotype[i].pred] = True
                elif type(self.genotype[i]) is Neurone:
                    self.used_node[self.genotype[i].pred1] = True
                    self.used_node[self.genotype[i].pred2] = True

    def evaluate(self, inp: []) -> []:
        res = np.zeros((self.conf.inp + self.conf.node + self.conf.out), dtype="float64")
        res[:self.conf.inp] = inp
        for i in range(self.conf.inp, self.conf.inp + self.conf.node + self.conf.out):
            if self.used_node[i]:
                res[i] = self.genotype[i - self.conf.inp].evaluate(res)
        return res[-self.conf.out:]

    def clone(self):
        res = Genome(self.conf)
        res.genotype = [neuron.clone() for neuron in self.genotype]
        return res

    def mutate(self, nb_mutation: int):
        for _ in range(nb_mutation):
            self.compute_used_node()
            for _ in range(100):
                i = randint(0, len(self.genotype) - 1)
                self.genotype[i].mutate()
                if self.used_node[i]:
                    break
        self.compute_used_node()
        return self

    def breed(self, other):
        self.genotype = [self.genotype[i] if choice([True, False]) else other.genotype[i].clone() for i in range(len(self.genotype))]
        self.compute_used_node()
        return self

    def to_list_of_dict(self) -> []:
        return [node.to_dict() for node in self.genotype]

    def __eq__(self, other):
        if len(self.genotype) != len(other.genotype):
            return False
        for i in range(len(self.genotype)):
            if self.genotype[i] != other.genotype[i]:
                return False
        return True


    @staticmethod
    def from_list_of_dict(l: list, gc: GenomeConfig):
        res = Genome(gc)
        res.genotype = [Neurone.from_dict(node, gc) if node["type"] == "Neurone"
                        else OutputNeurone.from_dict(node, gc)
                        for node in l]
        res.compute_used_node()
        return res


class Population(object):
    list_genomes: [Genome] = []
    list_score: [] = []
    genome_config: GenomeConfig
    size: int
    keep: int
    breed: int
    muta_count: int

    def __init__(self, genome_config: GenomeConfig, size = 50, keep = 5, breed = None, muta_count = None):
        self.list_genomes = [Genome(genome_config) for _ in range(size)]
        self.list_scores = [None] * size
        self.genome_config = genome_config
        self.size = size
        self.keep = keep
        if breed is None:
            self.breed = (size - keep)//4
        else:
            self.breed = breed
        if muta_count is None:
            self.muta_count = genome_config.node//40
        else:
            self.muta_count = muta_count

    def keep_bests(self):
        zipres = [(self.list_scores[i],self.list_genomes[i]) for i in range(len(self.list_genomes))]
        bests = [t for t in sorted(zipres, key=lambda x: x[0])[-self.keep:]][::-1]
        self.list_scores = [i[0] for i in bests]
        self.list_genomes = [i[1] for i in bests]

    def next_gen(self):
        zipres = [(self.list_scores[i], self.list_genomes[i]) for i in range(len(self.list_genomes))]
        new_parents = [t for t in sorted(zipres, key=lambda x: x[0])[-self.keep:]][::-1]
        new_breed = [(None, choice(new_parents)[1].clone().breed(choice(new_parents)[1])) for _ in range(self.breed)]
        new_mutated = [(None, choice(new_parents)[1].clone().mutate(self.muta_count)) for _ in
                       range(self.size - self.keep - self.breed)]
        new = new_parents + new_breed + new_mutated
        self.list_scores = [i[0] for i in new]
        self.list_genomes = [i[1] for i in new]

    def serialize(self) -> []:
        tmp_list = []
        for i in range(len(self.list_genomes)):
            tmp_list.append((self.list_scores[i], self.list_genomes[i].to_list_of_dict()))
        return tmp_list

    def save(self, save_name: str, save_dir="./saves/") -> None:
        with open("{}{}.json".format(save_dir, save_name), "w+") as outfile:
            json.dump(self.serialize(), outfile)
