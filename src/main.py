
from processing.feature import ScaleNormalization
from processing.partition import get_adjacency_by_group, get_adjacency_by_distance
from sample.util import generate_n_action_sequence, filter_by_metadata, list_values_by_field_name, merge_actions
from dataset.pose_sequence import PoseSequenceDataset
from dataset.transform import SplitFramesAndLabels, UnfoldFeatures, ToSTGcn
from nn.layers import ST_GCN
import torch
from torch.utils.data import Dataset, DataLoader


import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from IPython.display import display
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import stats
from math import pi
from numpy import inf
import imageio
import sklearn
from copy import deepcopy

from genetic_algorithm import GeneticAlgorithm, initial_population, init_next_phase_population



population_size = 20
num_generations = 15
num_phase = 5

ga = GeneticAlgorithm(tournament_size = 2)

next_generation = initial_population(population_size)
previous_state = None

overall_fitness = []
overall_genomes = []
overall_models = []

for phase in range(num_phase):
    print('Phase {}'.format(phase))
    bests_fitness = []
    bests_genomes = []
    bests_models = []
    for gen in range(num_generations):
        next_generation, best_genome, best_model, best_fitness = ga.execute_generation(
            next_generation,
            previous_state=previous_state,
            verbose=False)
        bests_fitness.append(best_fitness)
        bests_genomes.append(best_genome)
        bests_models.append(best_model)
        print(best_fitness)

    bests_fitness = np.array(bests_fitness)
    best_phase_fitness = bests_fitness[bests_fitness.argmax()]
    best_phase_genome = bests_genomes[bests_fitness.argmax()]
    best_phase_model = bests_models[bests_fitness.argmax()]

    overall_fitness.extend(bests_fitness)
    overall_genomes.extend(bests_genomes)
    overall_models.extend(bests_models)

    previous_state=best_phase_model.state_dict()
    next_phase_population = init_next_phase_population( best_phase_genome , population_size)
    next_generation = deepcopy(next_phase_population)


best_overall_fitness = np.array(overall_fitness)
best_overall_fitness = overall_fitness[best_overall_fitness.argmax()]
best_overall_genome = overall_genomes[best_overall_fitness.argmax()]
best_overall_model = overall_models[best_overall_fitness.argmax()]

print(best_overall_fitness)
print(best_overall_genome)
torch.save(best_overall_model.state_dict(), 'model_to_train.pth')

