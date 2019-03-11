import json
from random import randint

_default_func=[
    lambda x,y: x+y,
    lambda x,y: x-y,
    lambda x,y: x*y,
    lambda x,y: x/y
]

class Neurone:
    def __init__(self, conf, col_id):
        self.col_id = col_id
        self.conf = conf
        self.pred1 = randint(0, conf.inp + (col_id)*conf.row -1)
        self.pred2 = randint(0, conf.inp + (col_id)*conf.row -1)
        self.func = conf.func[randint(0, len(conf.func) -1)]

    def evaluate(self, genome):
        if self.pred1 < self.conf.inp:
            val1 = genome[self.pred1]
        else:
            val1 = genome[self.pred1].evaluate(genome)
        if self.pred2 < self.conf.inp:
            val2 = genome[self.pred2]
        else:
            val2 = genome[self.pred2].evaluate(genome)
        return self.func(val1, val2)

    def mutate(self):
        rand = randint(0,2)
        if rand == 0:
            self.pred1 = randint(0, self.conf.inp + (self.col_id)*self.conf.row -1)
        elif rand==1:
            self.pred2 = randint(0, self.conf.inp + (self.col_id)*self.conf.row -1)
        else:
            self.func = self.conf.func[randint(0, len(self.conf.func) -1)]


class GenomeConfig:
    def __init__(self, inp, out, row = 1, col = 1, func = _default_func):
        self.inp = inp
        self.out = out
        self.row = row
        self.col = col
        self.func = func

class Genome:
    def __init__(self, conf):
        self.conf = conf
        self.genotype = [0] * conf.inp
        for c in range(conf.col):
            self.genotype += [neurone(conf, c) for _ in range(conf.row)]
        self.genotype += [randint(0, conf.inp + conf.row*conf.col -1) for _ in range(conf.out)]

    def evaluate(self,inp):
        for i in range(self.conf.inp):
            self.genotype[i] = inp[i]
        res = [0] * self.conf.out
        for i in range(self.conf.out):
            j = i + self.conf.inp + self.conf.row*self.conf.col
            if self.genotype[j] < self.conf.inp:
                res[i] = self.genotype[self.genotype[j]]
            else:
                res[i] = self.genotype[self.genotype[j]].evaluate(self.genotype)
        return res
            
    def mutate(self, nb_mutation):
        for _ in range(nb_mutation):
            i = randint(self.conf.inp, len(self.genotype) -1)
            if i < self.conf.inp + self.conf.row*self.conf.col:
                self.genotype[i].mutate()
            else:
                self.genotype[i] = randint(0, self.conf.inp + self.conf.row*self.conf.col -1)

    def evaluate(self, inputs: []) -> []:
        pass

    @classmethod
    def serialize(cls):
        pass

    @classmethod
    def deserialize(cls, e):
        pass


class Population(object):
    list_genomes = []
    genome_config = GenomeConfig

    def __init__(self, genome_config : GenomeConfig, size = 50):
        self.genome_config = genome_config
        for i in range(size):
            g = Genome(genome_config)
            self.list_genomes.append(g)

    def serialize(self) -> str:
        tmp_list = []
        for g in self.list_genomes:
            tmp_list.append(g.serialize())
        return json.dumps(self.list_genomes)

    def make_them_play(self, nb_runs = 5, keep = 5):
        pass

    @classmethod
    def deserialize(cls, str, genome_config) :
        tmp_list = json.loads(str)
        new_pop = Population(genome_config)
        for e in tmp_list:
            g = Genome.deserialize(e)
            new_pop.list_genomes.append(g)
        return new_pop
