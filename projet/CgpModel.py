import json
from random import randint

_default_func = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y
]


class GenomeConfig:
    def __init__(self, inp: int, out: int, row=1, col=1, func=None):
        if func is None:
            func = _default_func
        self.inp = inp
        self.out = out
        self.row = row
        self.col = col
        self.func = func


class Neurone:
    def __init__(self, genome_config: GenomeConfig, col_id: int):
        self.col_id = col_id
        self.conf = genome_config
        self.pred1 = randint(0, genome_config.inp + col_id * genome_config.row - 1)
        self.pred2 = randint(0, genome_config.inp + col_id * genome_config.row - 1)
        self.func = randint(0, len(genome_config.func) - 1)

    def evaluate(self, genome: []):
        if self.pred1 < self.conf.inp:
            val1 = genome[self.pred1]
        else:
            val1 = genome[self.pred1].evaluate(genome)
        if self.pred2 < self.conf.inp:
            val2 = genome[self.pred2]
        else:
            val2 = genome[self.pred2].evaluate(genome)
        return self.conf.func[self.func](val1, val2)

    def clone(self):
        res = Neurone(self.conf, self.col_id)
        res.pref1 = self.pred1
        res.pref2 = self.pred2
        res.func = self.func
        return res

    def mutate(self):
        rand = randint(0, 2)
        if rand == 0:
            self.pred1 = randint(0, self.conf.inp + self.col_id * self.conf.row - 1)
        elif rand == 1:
            self.pred2 = randint(0, self.conf.inp + self.col_id * self.conf.row - 1)
        else:
            self.func = self.conf.func[randint(0, len(self.conf.func) - 1)]

    def serialize(self) -> dict:
        res = dict()
        res["col_id"] = self.col_id
        res["pred1"] = self.pred1
        res["pred2"] = self.pred2
        res["func"] = self.func
        return res

def deserialize_neurone(genome_config: GenomeConfig, d: dict) -> Neurone:
    res = Neurone(genome_config, d["col_id"])
    res.pred1 = d["pred1"]
    res.pred2 = d["pred2"]
    res.func = d["func"]
    return res

class Genome:
    def __init__(self, genome_config: GenomeConfig):
        self.conf = genome_config
        self.genotype = [0] * genome_config.inp
        for c in range(genome_config.col):
            self.genotype += [Neurone(genome_config, c) for _ in range(genome_config.row)]
        self.genotype += [randint(0, genome_config.inp + genome_config.row * genome_config.col - 1) for _ in
                          range(genome_config.out)]

    def evaluate(self, inp: []) -> []:
        for i in range(self.conf.inp):
            self.genotype[i] = inp[i]
        res = [0] * self.conf.out
        for i in range(self.conf.out):
            j = i + self.conf.inp + self.conf.row * self.conf.col
            if self.genotype[j] < self.conf.inp:
                res[i] = self.genotype[self.genotype[j]]
            else:
                res[i] = self.genotype[self.genotype[j]].evaluate(self.genotype)
        return res
        # Comment prendre la décision ? CGP est il mieux opti en renvoyant une liste de nombres ? Une liste de bits ?
        # Un seul nombre ?

    def clone(self):
        res = Genome(self.conf)
        res.genotype = [0] * res.conf.inp
        res.genotype += [ self.genotype[res.conf.inp + i].clone() for i in range(res.conf.row * res.conf.col)]
        res.genotype += [ self.genotype[res.conf.inp + res.conf.row * res.conf.col + i] for i in range(res.conf.out)]
        return res

    def mutate(self, nb_mutation: int):
        for _ in range(nb_mutation):
            i = randint(self.conf.inp, len(self.genotype) - 1)
            if i < self.conf.inp + self.conf.row * self.conf.col:
                self.genotype[i].mutate()
            else:
                self.genotype[i] = randint(0, self.conf.inp + self.conf.row * self.conf.col - 1)
        return self

    def serialize(self) -> str:
        res = [ node if type(node) is int else node.serialize() for node in self.genotype ]
        return json.dumps(res)


def deserialize_genome(gc: GenomeConfig, s: str) -> Genome:
    l = json.loads(s)
    res = Genome(gc)
    res.genotype = [ node if type(node) is int else deserialize_neurone(gc, node) for node in l ]
    return res


class Population(object):
    list_genomes: [Genome] = []
    genome_config = GenomeConfig

    def __init__(self, genome_config: GenomeConfig, size=50):
        self.genome_config = genome_config
        for i in range(size):
            g = Genome(genome_config)
            self.list_genomes.append(g)

    def serialize(self) -> str:
        tmp_list = []
        for g in self.list_genomes:
            tmp_list.append(g.serialize())
        return json.dumps(self.list_genomes)


def deserialize_population(s: str, genome_config: GenomeConfig) -> Population:
    tmp_list = json.loads(s)
    new_pop = Population(genome_config)
    for e in tmp_list:
        g = deserialize_genome(e)
        new_pop.list_genomes.append(g)
    return new_pop


def generate_population_from(bests: [Genome], child_count: int, muta_count=10):
    # give birth bests.size - pop_size from bests winner by mutation
    res = bests
    res += [ bests[randint(0, len(bests) - 1)].clone().mutate(muta_count) for _ in range(child_count - len(bests))]
    return res