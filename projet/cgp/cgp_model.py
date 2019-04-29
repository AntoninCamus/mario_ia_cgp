import json
from random import randint, choice

import numpy as np

from cgp.cgp_functions import UNARY_FUNCTIONS, BINARY_FUNCTIONS, UNARY_REDUCERS

_binary_func = BINARY_FUNCTIONS
_unary_func = UNARY_FUNCTIONS
_unary_reduce_func = UNARY_REDUCERS
_binary_reduce_func = []


class GenomeConfig:
    def __init__(
            self,
            inp: int,
            out: int,
            node=1,
            binary_func=None,
            unary_func=None,
            binary_reduce_func=None,
            unary_reduce_func=None,
    ):
        if binary_func is None:
            binary_func = _binary_func
        if unary_func is None:
            unary_func = _unary_func
        if binary_reduce_func is None:
            binary_reduce_func = _binary_reduce_func
        if unary_reduce_func is None:
            unary_reduce_func = _unary_reduce_func
        self.inp = inp
        self.out = out
        self.node = node
        self.binary_func = binary_func
        self.unary_func = unary_func
        self.binary_reduce_func = binary_reduce_func
        self.unary_reduce_func = unary_reduce_func


class BinaryNeurone:
    def __init__(self, genome_config: GenomeConfig, node_id: int):
        self.node_id = node_id
        self.conf = genome_config
        self.pred1 = randint(0, genome_config.inp + node_id - 1)
        self.pred2 = randint(0, genome_config.inp + node_id - 1)
        self.func = randint(0, len(genome_config.binary_func) - 1)

    def evaluate(self, data: [[float]]):
        val1 = data[self.pred1]
        val2 = data[self.pred2]
        return self.conf.binary_func[self.func](val1, val2)

    def clone(self):
        res = BinaryNeurone(self.conf, self.node_id)
        res.pred1 = self.pred1
        res.pred2 = self.pred2
        res.func = self.func
        return res

    def mutate(self):
        if randint(0, 100) < 5:
            return UnaryNeurone(self.conf, self.node_id)
        rand = randint(0, 2)
        if rand == 0:
            self.pred1 = randint(0, self.conf.inp + self.node_id - 1)
        elif rand == 1:
            self.pred2 = randint(0, self.conf.inp + self.node_id - 1)
        else:
            self.func = randint(0, len(self.conf.binary_func) - 1)
        return self

    def preds(self) -> [int]:
        return [self.pred1, self.pred2]

    def to_dict(self) -> dict:
        return {
            "type": "BinaryNeurone",
            "node_id": self.node_id,
            "pred1": self.pred1,
            "pred2": self.pred2,
            "func": self.func,
        }

    def __eq__(self, other):
        if not isinstance(other, BinaryNeurone):
            return False
        return (
                self.pred1 == other.pred1
                and self.pred2 == other.pred2
                and self.func == other.func
        )

    @staticmethod
    def from_dict(d: dict, genome_config: GenomeConfig):
        if d["type"] != "BinaryNeurone":
            return None
        res = BinaryNeurone(genome_config, d["node_id"])
        res.pred1 = d["pred1"]
        res.pred2 = d["pred2"]
        res.func = d["func"]
        return res


class UnaryNeurone:
    def __init__(self, genome_config: GenomeConfig, node_id: int):
        self.conf = genome_config
        self.node_id = node_id
        self.pred = randint(0, genome_config.inp + node_id - 1)
        self.func = randint(0, len(genome_config.unary_func) - 1)

    def evaluate(self, data: [[float]]):
        return self.conf.unary_func[self.func](data[self.pred])

    def clone(self):
        res = UnaryNeurone(self.conf, self.node_id)
        res.pred = self.pred
        res.func = self.func
        return res

    def mutate(self):
        if randint(0, 100) < 5:
            return BinaryNeurone(self.conf, self.node_id)
        if randint(0, 1):
            self.pred = randint(0, self.conf.inp + self.node_id - 1)
        else:
            self.func = randint(0, len(self.conf.unary_func) - 1)
        return self

    def preds(self) -> [int]:
        return [self.pred]

    def to_dict(self) -> dict:
        return {
            "type": "UnaryNeurone",
            "node_id": self.node_id,
            "pred": self.pred,
            "func": self.func,
        }

    def __eq__(self, other):
        if not isinstance(other, UnaryNeurone):
            return False
        return self.pred == other.pred and self.func == other.func

    @staticmethod
    def from_dict(d: dict, genome_config: GenomeConfig):
        if d["type"] != "UnaryNeurone":
            return None
        res = UnaryNeurone(genome_config, d["node_id"])
        res.pred = d["pred"]
        res.func = d["func"]
        return res


class BinaryOutputNeurone:
    def __init__(self, genome_config: GenomeConfig):
        self.conf = genome_config
        self.pred1 = randint(0, genome_config.inp + genome_config.node - 1)
        self.pred2 = randint(0, genome_config.inp + genome_config.node - 1)
        self.func = randint(0, len(genome_config.unary_reduce_func) - 1)

    def evaluate(self, data: [[float]]):
        val1 = data[self.pred1]
        val2 = data[self.pred2]
        return self.conf.binary_reduce_func[self.func](val1, val2)

    def clone(self):
        res = BinaryOutputNeurone(self.conf)
        res.pred1 = self.pred1
        res.pred2 = self.pred2
        res.func = self.func
        return res

    def mutate(self):
        if randint(0, 100) < 5:
            return UnaryOutputNeurone(self.conf)
        i = randint(0, 2)
        if i == 0:
            self.pred1 = randint(0, self.conf.inp + self.conf.node - 1)
        elif i == 1:
            self.pred2 = randint(0, self.conf.inp + self.conf.node - 1)
        else:
            self.func = randint(0, len(self.conf.binary_reduce_func) - 1)
        return self

    def preds(self) -> [int]:
        return [self.pred1, self.pred2]

    def to_dict(self) -> dict:
        return {
            "type": "BinaryOutputNeurone",
            "pred1": self.pred1,
            "pred2": self.pred2,
            "func": self.func,
        }

    def __eq__(self, other):
        if not isinstance(other, BinaryOutputNeurone):
            return False
        return (
                self.pred1 == other.pred1
                and self.pred2 == other.pred2
                and self.func == other.func
        )

    @staticmethod
    def from_dict(d: dict, genome_config: GenomeConfig):
        if d["type"] != "BinaryOutputNeurone":
            return None
        res = BinaryOutputNeurone(genome_config)
        res.pred1 = d["pred1"]
        res.pred2 = d["pred2"]
        res.func = d["func"]
        return res


class UnaryOutputNeurone:
    def __init__(self, genome_config: GenomeConfig):
        self.conf = genome_config
        self.pred = randint(0, genome_config.inp + genome_config.node - 1)
        self.func = randint(0, len(genome_config.unary_reduce_func) - 1)

    def evaluate(self, data: []):
        return self.conf.unary_reduce_func[self.func](data[self.pred])

    def clone(self):
        res = UnaryOutputNeurone(self.conf)
        res.pred = self.pred
        res.func = self.func
        return res

    def mutate(self):
        # if randint(0,100) < 5:
        #   return BinaryOutputNeurone(self.conf)
        if randint(0, 1):
            self.pred = randint(0, self.conf.inp + self.conf.node - 1)
        else:
            self.func = randint(0, len(self.conf.unary_reduce_func) - 1)
        return self

    def preds(self) -> [int]:
        return [self.pred]

    def to_dict(self) -> dict:
        return {"type": "UnaryOutputNeurone", "pred": self.pred, "func": self.func}

    def __eq__(self, other):
        if not isinstance(other, UnaryOutputNeurone):
            return False
        return self.pred == other.pred and self.func == other.func

    @staticmethod
    def from_dict(d: dict, genome_config: GenomeConfig):
        if d["type"] != "UnaryOutputNeurone":
            return None
        res = UnaryOutputNeurone(genome_config)
        res.pred = d["pred"]
        res.func = d["func"]
        return res


class Genome:
    def __init__(self, genome_config: GenomeConfig):
        self.conf = genome_config
        self.genotype = [
            BinaryNeurone(genome_config, i)
            if randint(0, 1)
            else UnaryNeurone(genome_config, i)
            for i in range(genome_config.node)
        ]
        self.genotype += [
            UnaryOutputNeurone(genome_config) for _ in range(genome_config.out)
        ]
        # self.genotype += [BinaryOutputNeurone(genome_config) if randint(0, 1) else UnaryOutputNeurone(genome_config)
        #                  for _ in range(genome_config.out)]
        self.used_node = np.empty((), dtype=bool)
        self.compute_used_node()

    def compute_used_node(self):
        self.used_node = np.zeros(
            (self.conf.inp + self.conf.node + self.conf.out,), dtype=bool
        )
        for i in range(self.conf.node + self.conf.out - 1, -1, -1):
            if (
                    type(self.genotype[i]) is BinaryOutputNeurone
                    or type(self.genotype[i]) is UnaryOutputNeurone
            ):
                self.used_node[i + self.conf.inp] = True
            if self.used_node[i + self.conf.inp]:
                for pred_idx in self.genotype[i].preds():
                    self.used_node[pred_idx] = True

    def evaluate(self, inp: [[]]) -> []:
        res = list(inp) + [None] * self.conf.node
        res[: self.conf.inp] = inp
        for i in range(self.conf.inp, self.conf.inp + self.conf.node):
            if self.used_node[i]:
                try:
                    res[i] = self.genotype[i - self.conf.inp].evaluate(res)
                    if np.isnan(res[i]).any() or np.isinf(res[i]).any():
                        raise ArithmeticError
                except Exception as e:
                    print(
                        "Error",
                        e,
                        "with node",
                        self.genotype[i - self.conf.inp],
                        self.genotype[i - self.conf.inp].func,
                        "output is",
                        res[i],
                    )
                    res[i] = 1
                    for j in self.genotype[i - self.conf.inp].preds():
                        print(
                            "input",
                            self.genotype[j - self.conf.inp],
                            self.genotype[j - self.conf.inp].func,
                            "with value",
                            res[j],
                        )

        return np.array(
            [
                self.genotype[i].evaluate(res)
                for i in range(self.conf.node, self.conf.node + self.conf.out)
            ]
        )

    def clone(self):
        res = Genome(self.conf)
        res.genotype = [neuron.clone() for neuron in self.genotype]
        res.compute_used_node()
        return res

    def mutate(self, nb_mutation: int):
        for _ in range(nb_mutation):
            self.compute_used_node()
            for _ in range(100):
                i = randint(0, len(self.genotype) - 1)
                self.genotype[i] = self.genotype[i].mutate()
                if self.used_node[i]:
                    break
        self.compute_used_node()
        return self

    def breed(self, other):
        self.genotype = [
            self.genotype[i] if choice([True, False]) else other.genotype[i].clone()
            for i in range(len(self.genotype))
        ]
        self.compute_used_node()
        return self

    def to_list_of_dict(self) -> []:
        return [node.to_dict() for node in self.genotype]

    def __eq__(self, other):
        if len(self.genotype) != len(other.genotype):
            return False
        for i in range(len(self.genotype)):
            if self.genotype[i] != other.genotype[i] and self.used_node[i + self.conf.inp] and other.used_node[i + other.conf.inp]:
                return False
        return True

    @staticmethod
    def from_list_of_dict(l: list, gc: GenomeConfig):
        res = Genome(gc)
        neurone_types = [
            UnaryNeurone,
            BinaryNeurone,
            UnaryOutputNeurone,
            BinaryOutputNeurone,
        ]

        def not_none(x):
            for i in x:
                if i is not None:
                    return i

        res.genotype = [
            not_none([Neurone.from_dict(node, gc) for Neurone in neurone_types])
            for node in l
        ]
        res.compute_used_node()
        return res


class Population(object):
    def __init__(
            self, genome_config: GenomeConfig, size=50, keep=5, breed=None, muta_count=None
    ):
        self.list_genomes = [Genome(genome_config) for _ in range(size)]
        self.list_scores = [None] * size
        self.genome_config = genome_config
        self.size = size
        self.keep = keep
        if breed is None:
            self.breed = (size - keep) // 4
        else:
            self.breed = breed
        if muta_count is None:
            self.muta_count = genome_config.node // 40
        else:
            self.muta_count = muta_count

    def keep_bests(self):
        zipped_result = zip(self.list_scores, self.list_genomes)
        bests = [t for t in sorted(zipped_result, key=lambda x: x[0])[-self.keep:]][::-1]
        self.list_scores = [i[0] for i in bests]
        self.list_genomes = [i[1] for i in bests]

    def next_gen(self):
        zipped_result = zip(self.list_scores, self.list_genomes)
        new_parents = [t for t in sorted(zipped_result, key=lambda x: x[0])[-self.keep:]][::-1]
        new_breed = [
            (None, choice(new_parents)[1].clone().breed(choice(new_parents)[1]))
            for _ in range(self.breed)
        ]
        new_mutated = [
            (None, choice(new_parents)[1].clone().mutate(self.muta_count))
            for _ in range(self.size - self.keep - self.breed)
        ]
        new = new_parents + new_breed + new_mutated
        self.list_scores = [i[0] for i in new]
        self.list_genomes = [i[1] for i in new]

    def to_list(self) -> [float, dict]:
        tmp_list = []
        for i in range(len(self.list_genomes)):
            tmp_list.append(
                (self.list_scores[i], self.list_genomes[i].to_list_of_dict())
            )
        return tmp_list

    def serialize(self) -> [float, str]:
        return [(e[0], json.dumps(e[1])) for e in self.to_list()]

    def save(self, save_name: str, save_dir="./saves/") -> None:
        with open("{}{}.json".format(save_dir, save_name), "w+") as outfile:
            json.dump(self.serialize(), outfile)
