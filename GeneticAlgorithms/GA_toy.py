import numpy as np
import random
import copy

class Gene():
    def __init__(self, gene_seed):
        if isinstance(gene_seed, int):
            self.genome = np.random.binomial(1, 0.5, gene_seed)
        else:
            self.genome = np.copy(gene_seed)

        self.gene_size = len(self.genome)

    def mutate(self):
        """ YOUR CODE HERE!"""
        ind = random.randint(0, len(self.genome) - 1)
        if self.genome[ind] == 1:
            self.genome[ind] = 0
        else:
            self.genome[ind] = 1

    def evaluate_fitness(self, target):
        """ Lower fitness is better. Perfect fitness should equal 0"""
        """ YOUR CODE HERE!"""
        fitness = len(self.genome)
        for i in range(len(self.genome)):
            if self.genome[i] == target[i]:
                fitness -= 1
        return fitness


class GeneticAlgorithm():
    def __init__(self, gene_size, population_size, target):
        self.gene_size = gene_size
        self.pop_size = population_size
        self.target = target
        self.parents = []

        self.gene_pool = [[np.nan, Gene(self.gene_size)] for _ in range(self.pop_size )]

    def evaluate_population(self):
        """ Evaluates the fitness of ever genome. If the best fitness is 0
            the function returns True, signalling that the optimization is done.
        """
        min_fitness = np.inf
        for gene in self.gene_pool:
            fitness = gene[1].evaluate_fitness(self.target)
            if min_fitness > fitness:
                min_fitness = fitness
            gene[0] = fitness

        if min_fitness == 0:
            return True
        return False

    def select_parents(self, num_parents):
        """ Function that selects num_parents from the population."""
        """ YOUR CODE HERE!"""
        self.gene_pool.sort(key=lambda x: x[0])
        self.parents = copy.deepcopy(self.gene_pool[:num_parents])

    def produce_next_generation(self):
        """ Function that creates the next generation based on parents."""
        """ YOUR CODE HERE!"""
        for parent in self.parents:
            #print "Before: " + str(parent[0])
            parent[1].mutate()
            parent[0] = parent[1].evaluate_fitness(self.target)
            #print "After: " + str(parent[0])
            for i in range(len(self.gene_pool)):
                if parent[0] < self.gene_pool[i][0]:
                    #print str(parent[0]) + " was less than " + str(self.gene_pool[i][0])
                    self.gene_pool[i] = parent
                    break
                #else:
                    #print str(parent[0]) + " was NOT less than " + str(self.gene_pool[i][0])

    def run(self):
        done = False
        i = 1
        while not done :
            done = self.evaluate_population()
            self.select_parents(int(self.pop_size/10)+1)

            if i % 5 == 0 or done:
                print("Generation:", i)
                print("Population:")
                for gene in self.gene_pool:
                    print("\tfit:", gene[0], gene[1].genome)
                print("Parents:")
                for parent in self.parents:
                    print("\tfit:", parent[0], parent[1].genome)
                print()

            self.produce_next_generation()
            i += 1

pop_size = 10
gene_size = 20
target = np.zeros(gene_size)

GA = GeneticAlgorithm(gene_size, pop_size, target)
GA.run()

print('Done')