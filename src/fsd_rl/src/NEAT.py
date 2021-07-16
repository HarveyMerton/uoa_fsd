#!/usr/bin/env python2.7

import numpy as np
import os
#import matplotlib.pyplot as plt
from collections import defaultdict, deque
#import networkx as nx
import math
import copy
from airl.gail_airl_ppo.utils import plot_results_rqt

"""
The NEAT algorithm:
Implementation by Chenrong Lu
NEAT paper: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
Original author: Kenneth O.Stanley, Risto Miikkulainen
"""

"""
Global Gene table stores the global 
innovation number for the genes(Connections)
"""


class GlobalGeneTable:
    def __init__(self, percept_size, out_size):
        self._innovation_table = defaultdict(
            lambda: None)  # Innovation table is a reverse mapping between connections to its innovation number
        self._node_innovation_table = defaultdict(lambda: None)
        self._node_innov_num = percept_size + out_size
        self._innov_num = 0
        for n in range(percept_size):
            for o in range(out_size):
                self._innovation_table[str(n) + "-" + str(o)] = self._innov_num
                self._innov_num += 1

    def get_innov_num(self, InOut):
        innov = self._innovation_table[InOut]
        if (innov != None):
            return innov
        else:
            self._innovation_table[InOut] = self._innov_num
            self._innov_num += 1
            return self._innovation_table[InOut]

    def get_node_innov_num(self, InOut):
        # New hidden nodes are defined by their edges
        node_innov = self._node_innovation_table[InOut]
        if (node_innov != None):
            return node_innov
        else:
            self._node_innovation_table[InOut] = self._node_innov_num
            self._node_innov_num += 1
            return self._node_innovation_table[InOut]


class Genome:
    def __init__(self, IN, OUT, WEIGHT, ENABLED):
        self._In = IN
        self._Out = OUT
        self._Weight = WEIGHT
        self._Enabled = ENABLED
        self.computed = False

    def get_connection(self):
        return (self._In, self._Out)

    def set_weight(self):
        pass

    def get_weight(self):
        return self._Weight

    def set_disabled(self):
        self._Enabled = False

    def set_enabled(self):
        self._Enabled = True

    def is_enabled(self):
        return self._Enabled

    def set_computed(self, value):
        self.computed = value

    def is_computed(self):
        return self.computed


class Agent:
    def __init__(self, percept_size, out_size):
        # S = Sensor
        # O = Output
        # H = Hidden
        self.WEIGHTAGNOSTIC = False
        self.percept_size = percept_size
        self.out_size = out_size
        self._NodeGenes = defaultdict(lambda: None)
        for n in range(self.percept_size + self.out_size):
            self._NodeGenes[n] = 0
        # [NodeGene(n,"S",0) for n in range(self.percept_size)] +[NodeGene(self.percept_size+o,"O",0) for o in range(self.out_size)]
        self._ConnectGenes = {}
        self._connections = defaultdict(lambda: [])
        for n in range(self.percept_size):
            for o in range(self.out_size):
                # Here we store the connections in a dict,
                # And store an array to reference to the stored
                # connections via a key ->the innovation number of the Genome
                # Start with no connections
                if (np.random.rand() < 1):
                    self._ConnectGenes[n * self.out_size + o] = Genome(n, self.percept_size + o, np.random.choice(
                        [-1, 1]) if self.WEIGHTAGNOSTIC else np.random.normal(), True)
                else:
                    self._ConnectGenes[n * self.out_size + o] = Genome(n, self.percept_size + o, np.random.choice(
                        [-1, 1]) if self.WEIGHTAGNOSTIC else np.random.normal(), False)
                self._connections[n].append(n*self.out_size+o)
        # self.show_graph()

    def set_fitness(self, fitness):
        self._fitness = fitness

    def get_fitness(self):
        return self._fitness

    def forward(self, observation):
        """
        Forward propagate starting from percept nodes,
        and then to each of the nodes connected to the percept nodes
        """
        queue = deque([])
        computed_nodes = []
        for n in range(self.percept_size):
            queue.append(n)
            self._NodeGenes[n] += observation.flatten()[n]

        #print(observation)
        while (True):
            # Start propagating from percept nodes
            if (len(queue) > 0):
                next = queue.popleft()
            else:
                break
            #print(self._connections)
            # For each connection we traverse the graph
            for innov_num in self._connections[next]:
                if (self._ConnectGenes[innov_num].is_enabled() and self._ConnectGenes[
                    innov_num].is_computed() == False):
                    self._NodeGenes[self._ConnectGenes[innov_num]._Out] += self._NodeGenes[next] * self._ConnectGenes[
                        innov_num]._Weight
                    self._ConnectGenes[innov_num].set_computed(True)
                    connection_reach = self._ConnectGenes[innov_num]._Out
                    if (connection_reach not in computed_nodes and connection_reach not in queue):
                        queue.append(self._ConnectGenes[
                                         innov_num]._Out)  # Append the next skirt to the graph if the node is not traversed
                else:
                    pass
            computed_nodes.append(next)
        # print(np.array([self._NodeGenes[self.percept_size+o].value for o in range(self.out_size)]))
        # out = np.argmax(np.array([self._NodeGenes[self.percept_size+o] for o in range(self.out_size)]))
        out = np.array([self._NodeGenes[self.percept_size + o] for o in range(self.out_size)])
        # print("NODE GENES: {}".format(self._NodeGenes[12]))
        for n in self._ConnectGenes:
            self._ConnectGenes[n].set_computed(False)
        for n in (self._NodeGenes):
            if (self._NodeGenes[n] != None):
                self._NodeGenes[n] = 0
        # for n in (self.percept_size,self.percept_size+self.out_size):
        #    self._NodeGenes[n] = 0

        out = out/15
        print("OUT: {}".format(out))
        return out

    def mutate_add_connection(self):
        # Choose two nodes in our nodes
        valid_start_nodes = []
        valid_end_nodes = []
        for key in self._NodeGenes:
            # Only choose from percetps or hidden to start
            if (key < self.percept_size or key >= self.percept_size + self.out_size):
                if (self._NodeGenes[key] != None):
                    valid_start_nodes.append(key)
            # Only choose from hidden or end, to connect
            elif (key >= self.percept_size):
                valid_end_nodes.append(key)

        choice_start = np.random.choice(valid_start_nodes, size=1, replace=False)
        choice_end = np.random.choice(valid_end_nodes, size=1, replace=False)
        nodes_choice = (choice_start[0], choice_end[0])
        # Get our next connection's global innovation number
        # If the connection is new, we store it as a new innovation number, otherwise
        # We should have a correspondant entry already
        next_innov_num = GlobalVar.ggt.get_innov_num(str(nodes_choice[0]) + "-" + str(nodes_choice[1]))

        # Append to our Node connections
        self._connections[nodes_choice[0]].append(next_innov_num)  # Add another gene block with next innov number
        # Append to our Genes
        self._ConnectGenes[next_innov_num] = Genome(nodes_choice[0], nodes_choice[1], np.random.choice(
            [-1, 1]) if self.WEIGHTAGNOSTIC else np.random.normal(), True)
        # self.show_graph()

    def mutate_add_node(self):
        # Choose a connection from our connections
        valid_connections = []
        for key in self._ConnectGenes.keys():
            if (self._ConnectGenes[key].is_enabled()):
                valid_connections.append(key)
        if (len(valid_connections) == 0):
            return
            # Choose from our enabled connections
        connection_choice = np.random.choice(valid_connections, size=1, replace=False)

        # Disable it and remove it from our node connections
        self._ConnectGenes[connection_choice[0]].set_disabled()
        # Get it's connections
        connection = self._ConnectGenes[connection_choice[0]].get_connection()

        """
        Here we get the innov number for the new node we are appending.
        This new number can be new or already existing in the entry 
        of the global table.
        We need to ensure that two nodes with the same number in two 
        different genes should mean the same thing.
        We do this by defining the nodes to the connection which they are
        originated from.
        Since there could be the situation with Another connection being
        added after it has been disabled.

        Example:
        [node1] ->[node2]   -----> [node1]->[node3]->[node2] -----> [node1]->[node3]->[node2]
                                                                       ------------------^
        In which case if a node [node4] were to be added, we would like it to have a unique                                                      
        identifier.
        """
        node_unique_distinguisher = 0
        """
        while(True):
            occupied = False
            if(node_unique_distinguisher==0):
                next_node_num = GlobalVar.ggt.get_node_innov_num(str(connection[0])+"-"+str(connection[1]))
            else:
                next_node_num = GlobalVar.ggt.get_node_innov_num(str(connection[0])+"-"+str(connection[1])+"-"+str(node_unique_distinguisher))
            for node in self._NodeGenes:
                if(node==next_node_num):                    
                    occupied = True
            if(occupied==False):
                #If we don't currently have a connection with same innov value we are good to assign the next node num
                break
            else:
                #Otherwise we keep checking
                node_unique_distinguisher +=1
        """
        next_node_num = GlobalVar.ggt.get_node_innov_num(str(connection[0]) + "-" + str(connection[1]))
        """
        We add the connections to our connection Genes
        """
        # First Connection, from In -> newnode
        next_innov_num1 = GlobalVar.ggt.get_innov_num(str(connection[0]) + "-" + str(next_node_num))
        # Second Connection, from newnode -> Out
        next_innov_num2 = GlobalVar.ggt.get_innov_num(str(next_node_num) + "-" + str(connection[1]))

        # Add another node to our nodes
        self._NodeGenes[next_node_num] = 0

        # Add a connection from In->newnode
        self._connections[connection[0]].append(next_innov_num1)
        # Add a connection from newNode->Out
        self._connections[next_node_num].append(next_innov_num2)

        # Add a connection from In->newnode
        self._ConnectGenes[next_innov_num1] = Genome(connection[0], next_node_num,
                                                     -1 if self.WEIGHTAGNOSTIC else np.random.normal(), True)
        # Add a connection from newnode->Out
        self._ConnectGenes[next_innov_num2] = Genome(next_node_num, connection[1],
                                                     -1 if self.WEIGHTAGNOSTIC else np.random.normal(), True)

    def crossover(self, partner, equality=False):
        # -----------------------------------
        # Align genes,choose randomly for
        # ones which are aligned, and inherit
        # disjoint and excess genes from more fit
        # parent
        for gene in self._ConnectGenes:
            if (gene not in partner._ConnectGenes.keys()):
                # Disjoint or access
                pass
            else:
                # Otherwise we choose between us and partner randomly
                if (np.random.rand() < 0.5):
                    self._ConnectGenes[gene] = copy.copy(partner._ConnectGenes[gene])
        if (equality):
            for node in partner._NodeGenes:
                if (partner._NodeGenes[node] != None):
                    if (self._NodeGenes[node] == None):
                        self._NodeGenes[node] = copy.copy(
                            partner._NodeGenes[node])  # we don't need to copy here since value is integer
            for gene in partner._ConnectGenes:
                if (gene not in self._ConnectGenes.keys()):
                    # parnter disjoint or excess
                    self._ConnectGenes[gene] = copy.copy(partner._ConnectGenes[gene])

    def show_graph(self):
        G = nx.Graph()
        color_map = []
        edges = []
        added = []
        for n in range(self.percept_size):
            G.add_node(n)
            color_map.append("red")
        for o in range(self.out_size):
            G.add_node(o)
            color_map.append("blue")
        for n in self._ConnectGenes:
            Genome = self._ConnectGenes[n]
            connection = Genome.get_connection()
            if (connection[1] not in added and connection[1] >= self.percept_size + self.out_size):
                G.add_node(o)
                color_map.append("grey")
                added.append(connection[1])
            if (connection[0] not in added and connection[0] >= self.percept_size + self.out_size):
                G.add_node(o)
                color_map.append("grey")
                added.append(connection[0])
            if (Genome.is_enabled()):
                G.add_edge(*connection, color='black', weight=Genome.get_weight())
        pos = nx.circular_layout(G)
        edges = G.edges()
        colors = [G[u][v]['color'] for u, v in edges]
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw(G, pos, edges=edges, with_labels=True, node_color=color_map, edge_color=colors, width=weights)
        # nx.draw(G,with_labels=True,font_weight="bold")
        plt.show()


class NEAT:
    def __init__(self, percept_size, out_size, popsize):
        self.agent_percept_size = percept_size
        self.agent_out_size = out_size
        self.species = []
        self.MUTATION_RATE = 0.4
        self.c1 = 1  # Excess
        self.c2 = 1  # Disjoint
        self.c3 = 0.4  # Weight difference
        self.d_t = 3  # Explicit fitness sharing threshhold
        self.fitness_scaling_factor = 1.5
        self.compatibility_threshhold = 1
        self.initial_species_num = 1
        self.leeway = 0.1
        self.INITIATED = False
        self.USESPECIATION = False
        self.POPSIZE = popsize
        self.population = self.get_initial_population()

    def get_initial_population(self):
        return [Agent(self.agent_percept_size, self.agent_out_size) for _ in range(self.POPSIZE)]

    def crossover(self, agent1, agent2):
        pass

    def register_next(self, n, fitness):
        self.population[n].set_fitness(fitness)

    def get_population(self):
        return self.population

    def normalize(self, x):
        # Softmax normalization e^x_i/sum(e^x_i)
        print("Nat's X: {}".format(x))
        x = np.array(x)
        exp_sum = np.sum(np.exp(x))
        return np.exp(x) / exp_sum

    def evolve_next_population(self):
        new_pop = self.repopulate_speciate(self.population)
        self.population = new_pop
        return self.population

    def calculate_genetic_distance(self, agent1, agent2):
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        E = 0
        D = 0
        W = 0
        W_count = 0
        N1 = 0
        N2 = 0
        # ---------Iterate for agent 1 genes--------------
        for gene in agent1._ConnectGenes:
            N1 += 1
            if (gene not in agent2._ConnectGenes.keys()):
                if (gene > max(list(agent2._ConnectGenes.keys()))):
                    # Excess
                    E += 1
                else:
                    # Disjoint
                    D += 1
            else:
                # Weight difference
                W += np.power(agent1._ConnectGenes[gene]._Weight - agent2._ConnectGenes[gene]._Weight, 2)
                W_count += 1

        # ---------Iterate again for agent2----------
        for gene in agent2._ConnectGenes:
            N2 += 1
            if (gene not in agent1._ConnectGenes.keys()):
                if (gene > max(list(agent1._ConnectGenes.keys()))):
                    # Excess
                    E += 1
                else:
                    # Disjoint
                    D += 1
            else:
                # Weight difference
                W += np.power(agent1._ConnectGenes[gene]._Weight - agent2._ConnectGenes[gene]._Weight, 2)
                W_count += 1
        # See NEAT compatibility distance
        delta = (c1 * E) / max(N1, N2) + (c2 * D) / max(N1, N2) + c3 * (W / W_count)
        return delta

    # =======================================
    # --------------SPECIATION---------------
    # =======================================
    def get_explicit_fitness(self, creature, old_population):
        fitness_norm = 1  # start with 1 to avoid division by zero error
        for j in old_population:
            distance = self.calculate_genetic_distance(creature, j)
            if (distance > self.d_t):
                # We don't need to scale if distance is far away from
                # Others in the population
                fitness_norm += 0
            else:
                # If too close, we scale the fitness
                # This helps with not letting any
                # Single specie dominate too easily
                fitness_norm += self.fitness_scaling_factor
        c_fitness = creature.get_fitness()
        norm_fitness = c_fitness / fitness_norm
        return c_fitness, norm_fitness

    def repopulate_speciate(self, old_population):
        if (not self.INITIATED):
            initial_species = []
            # Randomly create some species
            initial_chosen = np.random.choice(len(old_population), size=self.initial_species_num, replace=False)
            for chosen_n in initial_chosen:
                initial_species.append({"creatures": [], "fitness": 0, "lastfitness": 0, "notgoodcount": 0,
                                        "rep": old_population[chosen_n]})
            biased_fitness = 0
            all_fitness = 0
            for n, creature in enumerate(old_population):
                # Compute normalized fitness through
                # explicit fitness sharing
                c_fitness, norm_fitness = self.get_explicit_fitness(creature, old_population)
                creature.set_fitness(c_fitness)
                biased_fitness += c_fitness
                all_fitness += norm_fitness
                # ------Appending Creature to Species------
                # Random choice
                specie = np.random.choice(self.initial_species_num)
                initial_species[specie]["creatures"].append(creature)
                initial_species[specie]["fitness"] += norm_fitness
            self.INITIATED = True
            species = initial_species
        else:  # IF INITIATED
            # ----------------------------------
            # put every other creature in species
            # ----------------------------------
            species = self.speicies
            biased_fitness = 0
            all_fitness = 0
            for n, creature in enumerate(old_population):
                # Compute normalized fitness through
                # explicit fitness sharing
                c_fitness, norm_fitness = self.get_explicit_fitness(creature, old_population)
                creature.set_fitness(c_fitness)
                biased_fitness += c_fitness
                all_fitness += norm_fitness
                # ------Appending Creature to Specie based on norm fitness------
                DONE = False
                for specie in range(len(species)):
                    rep = species[specie]["rep"]
                    distance = self.calculate_genetic_distance(creature, rep)
                    if (distance < self.compatibility_threshhold):
                        species[specie]["creatures"].append(creature)
                        species[specie]["fitness"] += norm_fitness
                        break
                    else:
                        # We create a new specie
                        species.append({"creatures": [creature], "fitness": 0, "lastfitness": 0, "notgoodcount": 0,
                                        "rep": creature})
        # ----------------------------------
        # -----We now have some species-----
        # -----Let's make some babies<3-----
        # ----------------------------------
        new_pop = []
        new_species = []
        pop_so_far = 0
        for specie in range(len(species)):
            # print("SPECIES CREATURES LEN: {}".format(len(species[specie]["creatures"])))
            # ------Bad performing species go extinct--------------
            if (species[specie]["notgoodcount"] == 15):
                all_fitness -= species[specie]["fitness"]
                continue  # We abandon the specie altogether, ie, don't append it to the new species
            if (species[specie]["fitness"] < species[specie]["lastfitness"] + self.leeway):
                # Ooo, not good not good
                species[specie]["notgoodcount"] += 1
            species[specie]["lastfitness"] = species[specie]["fitness"]

            # ------Otherwise we populate the next generation-------
            pop_fitness = [creature.get_fitness() for creature in species[specie]["creatures"]]
            try:
                population_share = math.floor(self.POPSIZE * species[specie]["fitness"] / (all_fitness))
            except Exception:
                population_share = 1
            for _ in range(int(population_share)):
                if (len(species[specie]["creatures"]) > 1):
                    parents = np.random.choice(species[specie]["creatures"], size=2, replace=False,
                                               p=self.normalize(pop_fitness))
                elif (len(species[specie]["creatures"]) > 0):
                    parents = (species[specie]["creatures"][0], species[specie]["creatures"][0])
                else:
                    break
                # ---------Cross over, the more fit is the dominant parent-----------
                fitness_p1 = parents[0].get_fitness()
                fitness_p2 = parents[1].get_fitness()
                if (fitness_p1 > fitness_p2):
                    child = copy.deepcopy(parents[0])
                    child.crossover(parents[1])
                elif (fitness_p2 > fitness_p1):
                    child = copy.deepcopy(parents[1])
                    child.crossover(parents[0])
                else:
                    # We are equal
                    child = copy.deepcopy(parents[0])
                    child.crossover(parents[1], equality=True)
                # ----------Mutation------------------------
                if (np.random.rand() < 0.8):
                    if (np.random.rand() > 0.5):
                        child.mutate_add_connection()
                    else:
                        child.mutate_add_node()
                # ---------Add to new population-----------
                # -------     Of the species     ----------
                new_pop.append(child)
                pop_so_far += 1
            # -----Add to new population to be evaluated-----
            species[specie]["creatures"] = []  # We have a new generation of new babies! Hooray
            species[specie]["fitness"] = 0  # We reset fitness
            new_species.append(species[specie])
        self.speicies = new_species
        while (len(new_pop) < self.POPSIZE):
            new_pop.append(np.random.choice(old_population, size=1)[0])
        return new_pop


class GlobalVar:
    ggt = None


if __name__ == "__main__":
    # Making gym environment
    import gym
    import time
    import rospy
    import fsd_env
    #import cv2

    # ROS
    rospy.init_node('fsd_gym', anonymous=True)

    env = gym.make("Fsd-v0")
    observation = env.reset()
    # Percept size and action size.
    #print(observation.shape)
    percept_size = len(observation.flatten())
    action_size = 1  # 2 For cartpole , 4 for bipedal walker
    GlobalVar.ggt = GlobalGeneTable(percept_size, action_size)

    NUM_GAMES = 10000
    POPSIZE = 100
    # Get initial population
    population = NEAT(percept_size, action_size, POPSIZE)
    # Roll out
    from tqdm import tqdm

    SOLVED = False
    cnt_episode = 0  # Count number of games played/generations passed
    avg_reward = 0
    for _ in range(NUM_GAMES):
        all_reward = 0
        best = -200
        best_agent = None

        for n, agent in tqdm(enumerate(population.get_population())):
            cum_reward = 0
            neg_count = 0
            while (True):
                # print(env.action_space.sample())
                # observation = cv2.resize(cv2.cvtColor(observation[30:200,:,:], cv2.COLOR_BGR2GRAY),(16,16)).flatten()/255
                action = agent.forward(observation.flatten())  # your agent here (this takes random actions)
                print("Action: {}".format(action))
                observation, reward, done, _ = env.step(action)
                cum_reward += reward
                if (reward < 0):
                    neg_count += 1
                else:
                    neg_count = 0
                if (neg_count > 100):
                    observation = env.reset()
                    done = True
                if done:
                    # if(SOLVED):
                    #    print(cum_reward)
                    if (cum_reward > best):
                        best_agent = agent
                        best = cum_reward
                    observation = env.reset()
                    break
            if (not SOLVED):
                population.register_next(n, cum_reward)
            all_reward += cum_reward
        viewbest = False
        if (viewbest):
            neg_count = 0
            while (True):
                action = best_agent.forward(observation.flatten())  # your agent here (this takes random actions)
                observation, reward, done, _ = env.step(action)
                if (reward < 0):
                    neg_count += 1
                else:
                    neg_count = 0
                if (neg_count > 100):
                    observation = env.reset()
                    break
                #env.render()
                if done:
                    observation = env.reset()
                    break
        with open("fsd_neat.txt", "a") as file:
            file.write(str(best) + "\n")
        avg_reward = all_reward / POPSIZE
        print("AVG: ", avg_reward)
        print("BEST: ", best)

        # Graph results
        plot_results_rqt(cnt_episode, avg_reward)

        # Update episode counter
        cnt_episode += 1

        if (SOLVED or all_reward / POPSIZE == 500):
            SOLVED = True
        else:
            population.evolve_next_population()
    #env.close()
