import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

from loader.taekwondo import TaekwondoLoader
from dataset.pose_sequence import PoseSequenceDataset
from sample.util import generate_n_action_sequence, filter_by_metadata, list_values_by_field_name, merge_actions
from dataset.transform import SplitFramesAndLabels, UnfoldFeatures, ToSTGcn
from evolutionary_model import EvolutionaryNet
from processing.feature import ScaleNormalization
from processing.partition import get_adjacency_by_group, get_adjacency_by_distance

from copy import deepcopy



layer_values = [2, 4, 8, 16, 32]
partition_values = ['distance', 'group']
graph_form_values = ['tree']
activation_values = ['relu', 'elu', 'tanh', 'leaky_relu']

activation_functions = {
    'relu': F.relu,
    'elu': F.elu,
    'tanh': F.tanh,
    'leaky_relu': F.leaky_relu
}

def generate_genome():
    g = []
    g.append(np.random.choice(layer_values))
    g.append(np.random.choice(partition_values))
    g.append(np.random.choice(graph_form_values))
    g.append(np.random.choice(activation_values))
    g.append(np.random.choice([True, False]))
    g.append(np.random.choice(layer_values))
    g.append(np.random.choice(partition_values))
    g.append(np.random.choice(graph_form_values))
    g.append(np.random.choice(activation_values))
    return g

def initial_population(population_size):
    population = []
    for i in range(population_size):
        population.append([generate_genome()])
    return population


def get_accuracy(out_tensor, labels_tensor):
    out_np = out_tensor.detach().cpu().numpy()
    out_np = np.array(np.argmax(out_np, axis=1))

    labels_np = np.array(labels_tensor.cpu().numpy())
    corrects = np.array(out_np == labels_np)

    result = np.sum(corrects) / (out_np.shape[0] * out_np.shape[1] * out_np.shape[2])
    return result



def tournament_selection(tournament_size, population, fitness):
    idx = np.random.permutation(len(population))
    sel_idx = idx[:tournament_size]
    # sel_fit = np.array([fitness[i] for i in sel_idx ])
    sel_fit = fitness[sel_idx]
    winner_idx = sel_idx[sel_fit.argmax()]
    return deepcopy(population[winner_idx])

def mutation(genome, p=0.05):
    r = np.random.random
    g = [None, None, None, None, None, None, None, None, None]
    g[0] = np.random.choice(layer_values) if r()<p else genome[0]
    g[1] = np.random.choice(partition_values) if r()<p else genome[1]
    g[2] = np.random.choice(graph_form_values) if r()<p else genome[2]
    g[3] = np.random.choice(activation_values) if r()<p else genome[3]
    g[4] = np.random.choice([True, False]) if r()<p else genome[4]
    g[5] = np.random.choice(layer_values) if r()<p else genome[5]
    g[6] = np.random.choice(partition_values) if r()<p else genome[6]
    g[7] = np.random.choice(graph_form_values) if r()<p else genome[7]
    g[8] = np.random.choice(activation_values) if r()<p else genome[8]
    return g

def crossover(parent_a, parent_b):
    child_a = [parent_a[0], parent_a[1], parent_a[2], parent_a[3], parent_b[4], parent_b[5], parent_b[6], parent_b[7], parent_b[8]]
    child_b = [parent_b[0], parent_b[1], parent_b[2], parent_b[3], parent_a[4], parent_a[5], parent_a[6], parent_a[7], parent_a[8]]
    return child_a, child_b

def print_population(population):
    for genome in population:
        print(genome)

def init_next_phase_population(selected_genome, population_size):
    population = []
    for i in range(population_size):
        new_genome = []
        new_genome.extend(selected_genome)
        new_genome.append(generate_genome())

        population.append(new_genome)
    return population

class GeneticAlgorithm():
    def __init__(self, tournament_size = 2):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 64
        self.tournament_size = tournament_size
        self.load_train_data()
        self.init_partition_modes()


    def init_partition_modes(self):
        self.partition_modes = {}

        partition_groups = {
            2: {'limb': 0, 'distance': 2},
            15: {'limb': 0, 'distance': 2},
            3: {'limb': 0, 'distance': 0},
            6: {'limb': 3, 'distance': 0},
            0: {'limb': 1, 'distance': 1},
            7: {'limb': 1, 'distance': 2},
            9: {'limb': 1, 'distance': 3},
            1: {'limb': 2, 'distance': 1},
            8: {'limb': 2, 'distance': 2},
            10: {'limb': 2, 'distance': 3},
            4: {'limb': 4, 'distance': 1},
            12: {'limb': 4, 'distance': 2},
            14: {'limb': 4, 'distance': 3},
            5: {'limb': 5, 'distance': 1},
            11: {'limb': 5, 'distance': 2},
            13: {'limb': 5, 'distance': 3}
        }

        group_adj, group_ks = get_adjacency_by_group(self.skeleton_model, partition_groups)
        self.partition_modes['group'] = {
            'ks': group_ks,
            'A': torch.from_numpy(group_adj).to(self.device, dtype=torch.float)
        }

        dist_adj, dist_ks = get_adjacency_by_distance(self.skeleton_model, partition_groups)
        self.partition_modes['distance'] = {
            'ks': dist_ks,
            'A': torch.from_numpy(dist_adj).to(self.device, dtype=torch.float)
        }


    def load_train_data(self):
        loader = TaekwondoLoader('../dataset')
        actions, labels, label_idx, skeleton_model = loader.load();
        print(labels)
        idx_label = { idx: key for (idx, key) in enumerate(labels)}

        scale = ScaleNormalization()
        for action in actions:
            scale(action)

        action_sequences = []

        for source in list_values_by_field_name(actions, 'source'):
            same_source_actions = filter_by_metadata(actions, 'source', source)
            merged_action = merge_actions(same_source_actions)
            action_sequences.append(merged_action)

        self.skeleton_model = skeleton_model
        self.sources = list_values_by_field_name(action_sequences, 'source')
        self.augmented_action_sequences = []

        for action_sequence in action_sequences:
            self.augmented_action_sequences = self.augmented_action_sequences + generate_n_action_sequence(action_sequence, 40)

        print('Not using unfold')
        self.transform = transforms.Compose([SplitFramesAndLabels(label_idx, ['position_xyz'], max_lenght=212),
                                    ToSTGcn()
                                ])



    def evaluate_fitness(self, model, epochs = 5):

        train_idxs = [0, 2, 3, 4, 5, 6, 7, 9]
        val_idxs = [1, 8]

        train_sources = [self.sources[i] for i in train_idxs]
        validation_sources = [self.sources[i] for i in val_idxs]
        train_sequences = filter_by_metadata(self.augmented_action_sequences, 'source', train_sources)
        validation_sequences = filter_by_metadata(self.augmented_action_sequences, 'source', validation_sources)
        train_dataset = PoseSequenceDataset(train_sequences, transform=self.transform)
        validation_dataset = PoseSequenceDataset(validation_sequences, transform=self.transform)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        model.train()

        for epoch in range(epochs):
            for samples in train_dataloader:
                data = samples['frames'].to(self.device, dtype=torch.float)
                labels = samples['labels'].to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                out = model(data)
                loss = F.cross_entropy(out, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        num_samples = 0
        accum = 0
        for samples in validation_dataloader:
            data = samples['frames'].to(self.device, dtype=torch.float)
            labels = samples['labels'].to(self.device, dtype=torch.long)
            out = model(data)
            accum = accum + get_accuracy(out, labels) * samples['frames'].shape[0]
            num_samples = num_samples + samples['frames'].shape[0]

        validation_accuracy = accum/num_samples
        return validation_accuracy


    def execute_generation(self, population_genomes, p_mut=0.1, p_cross=0.9, verbose=False, previous_state=None):
        population_size  = len(population_genomes)
        if verbose:
            print_population(population_genomes)
        population_fitness = []
        trained_models = []

        # Calculate fitness
        if verbose:
            print('Calculando fitness')
        for genome in population_genomes:
            model = EvolutionaryNet(3, 7, self.partition_modes, activation_functions, genome, previous_state=previous_state)
            fitness = self.evaluate_fitness(model)
            population_fitness.append(fitness)
            trained_models.append(model)

        population_fitness = np.array(population_fitness)
        best_genome = population_genomes[population_fitness.argmax()]
        best_model = trained_models[population_fitness.argmax()]

        # Select best individuals
        if verbose:
            print('Selecionando melhores indivíduos')
        selected_genomes = []
        for i in range(population_size):
            selected_genomes.append(tournament_selection(self.tournament_size, population_genomes, population_fitness))

        if verbose:
            print_population(selected_genomes)

        #Apply operators
        #Mutation
        if verbose:
            print('Aplicando operadores genéticos')
            print('Mutação')
        np.random.shuffle(selected_genomes)
        for i in range(population_size):
            selected_genomes[i][-1] = mutation(selected_genomes[i][-1], p=p_mut)

        if verbose:
            print_population(selected_genomes)

        if verbose:
            print('Cruzamento')
        for i in range(0, population_size, 2):
            if np.random.random() < p_cross:
                child_a, child_b = crossover(selected_genomes[i][-1], selected_genomes[i+1][-1])
                selected_genomes[i][-1] = deepcopy(child_a)
                selected_genomes[i+1][-1] = deepcopy(child_b)

        if verbose:
            print_population(selected_genomes)
            print('\nBest fitness: {}\n'.format(population_fitness.max()))
        return selected_genomes, best_genome, best_model, max(population_fitness)

