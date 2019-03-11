import json


class GenomeConfig(object):
    pass

class Genome:

    def __init__(self, config : GenomeConfig):
        pass
    # trinity
    def mutate(self, nb_mutation: int):
        pass

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

    @classmethod
    def deserialize(cls, str, genome_config) :
        tmp_list = json.loads(str)
        new_pop = Population(genome_config)
        for e in tmp_list:
            g = Genome.deserialize(e)
            new_pop.list_genomes.append(g)
        return new_pop
