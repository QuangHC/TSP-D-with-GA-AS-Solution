import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import time

class AS_Algorithm:
    def __init__(self, chromosome, distances, transportation_array, depot_point, n_ants, decay, alpha, beta, Ts, Ds) -> None:
        self.chromosome = chromosome
        self.truck = [i for i in range(len(chromosome)) if chromosome[i] == 0]
        self.drone = [i for i in range(len(chromosome)) if chromosome[i] == 1]
        self.distances = distances
        self.truckphmtrix = np.ones_like(distances) / len(distances) 
        self.dronephmtrix = np.ones_like(distances) / len(distances)
        self.all_inds = range(len(distances))
        self.transportation_array = transportation_array
        self.depot_point = depot_point
        self.n_ants = n_ants    
        self.decay = decay  
        self.alpha = alpha  
        self.beta = beta   
        self.Ts = Ts  
        self.Ds = Ds
        self.Tcost = self.distances * (1 / self.Ts) 
        self.Dcost = self.distances * (1 / self.Ds) 
        
    def run(self, n_iterations):
        best_path = None
        all_time_best_cost = float('inf')
        for i in range(n_iterations):
            ants = self.generate_ants()
            self.spread_pheromone(ants)
            # pheromone  = pheromone * decay
            current_best_path, current_best_cost = self.get_best_path(ants)
            if current_best_cost < all_time_best_cost:
                best_path = current_best_path
                all_time_best_cost = current_best_cost
        return best_path, self.fitness_measurement(best_path), self.type_Matrix(best_path)

    def generate_ants(self):
        ants = []
        for _ in range(self.n_ants):
            ant_path = self.generate_ant_path(self.depot_point)
            ants.append((ant_path, self.calculate_path_cost(ant_path)))
        return ants

    def generate_ant_path(self, start):
        ant_path = []
        unvisited = set(range(len(self.chromosome)))
        unvisited.remove(start)
        ant_path.append(start)
        while len(unvisited):
            new_unvisited = []
            next_city = None
            if self.transportation_array[ant_path[-1]] == 1:
                new_unvisited = list(
                    filter(lambda x: self.transportation_array[x] == 1 or self.transportation_array[x] == 3, unvisited))
            elif self.transportation_array[ant_path[-1]] == 2:
                new_unvisited = list(filter(lambda x: self.transportation_array[x] == 3, unvisited))
            elif self.transportation_array[ant_path[-1]] == 3:
                new_unvisited = list(filter(lambda x: self.transportation_array[x] != 1, unvisited))
            else:
                new_unvisited = list(
                    filter(lambda x: self.transportation_array[x] == 3 or self.transportation_array[x] == 2, unvisited))
            next_city = self.choose_next_city(ant_path[-1], new_unvisited if len(new_unvisited) else unvisited)
            if next_city is None:
                return [], 10**5
            ant_path.append(next_city)
            unvisited.remove(next_city)
        ant_path.append(start)
        return ant_path

    def choose_next_city(self, current_city, unvisited):
        pheromone_values = np.zeros_like(self.distances, dtype=float)
        for j in unvisited:
            if j in self.truck:
                p_tij = self.truckphmtrix[current_city][j]
                d_ij = self.distances[current_city][j]
                pheromone_values[current_city][j] = (p_tij ** self.alpha) * ((1 / d_ij) ** self.beta)
            else:
                p_dij = self.dronephmtrix[current_city][j]
                d_ij = self.distances[current_city][j]
                pheromone_values[current_city][j] = (p_dij ** self.alpha) * ((1 / d_ij) ** self.beta)

        arr_sum = 0
        for i in range(len(pheromone_values)):
            for j in range(len(pheromone_values[0])):
                arr_sum += pheromone_values[i][j]

        probabilities = pheromone_values * (1 / (arr_sum if arr_sum > 0 else 10**5))
        r = random.random()
        next_city = None
        cummulative_sum = 0
        for i in range(len(pheromone_values)):
            for j in range(len(pheromone_values[0])):
                cummulative_sum += probabilities[i][j]
                if cummulative_sum >= r and j != current_city and j in unvisited:
                    return j
        return next_city

    def spread_pheromone(self, ants):
        pheromone_truck_change = np.zeros_like(self.truckphmtrix)
        pheromone_drone_change = np.zeros_like(self.truckphmtrix)
        for ant_path, cost in ants:
            for i in range(len(ant_path) - 1):
                if ant_path[i + 1] in self.truck:
                    pheromone_truck_change[ant_path[i], ant_path[i + 1]] += 1 / cost
                    pheromone_truck_change[ant_path[i + 1], ant_path[i]] += 1 / cost
                elif ant_path[i+1] in self.drone:
                    pheromone_drone_change[ant_path[i], ant_path[i + 1]] += 1 / cost
                    pheromone_drone_change[ant_path[i + 1], ant_path[i]] += 1 / cost
        self.truckphmtrix = (1 - self.decay) * self.truckphmtrix + pheromone_truck_change
        self.dronephmtrix = (1 - self.decay) * self.dronephmtrix + pheromone_drone_change

    def get_best_path(self, ants):
        best_path = None
        best_cost = float('inf')
        for ant_path, cost in ants:
            if cost < best_cost:
                best_path = ant_path
                best_cost = cost
        return best_path, best_cost

    def calculate_path_cost(self, path):
        my_path = path[:]
        my_trans_matrix = self.type_Matrix(path)
        total_cost = 0
        i = 0
        while i < len(my_path) - 1:
            if my_trans_matrix[i + 1] == 3 or my_trans_matrix[i + 1] == 1:
                total_cost += self.Tcost[my_path[i]][my_path[i + 1]]
            elif my_trans_matrix[i + 1] == 4:
                total_cost += self.Dcost[my_path[i]][my_path[i + 1]] * 2
                my_path[i + 1] = my_path[i]
                my_trans_matrix[i + 1] = my_trans_matrix[i]
            elif my_trans_matrix[i + 1] == 2:
                start = next((j for j in reversed(range(i+1)) if my_trans_matrix[j] == 3), None)
                end = next((k for k in range(i+1, len(my_trans_matrix)) if my_trans_matrix[k] == 3), None)
                if start != None and end != None:
                    sub_truck_path = [my_path[start]] + [my_path[i] for i in range(start, end) if my_trans_matrix[i] == 1] + [my_path[end]]
                    sub_drone_path = [my_path[start]] + [my_path[i] for i in range(start, end) if my_trans_matrix[i] == 2 or my_trans_matrix[i] == 4] + [my_path[end]]
                    t_cost = sum(self.Tcost[sub_truck_path[i]][sub_truck_path[i + 1]] for i in range(len(sub_truck_path) - 1))
                    d_cost = 0
                    for i in range(len(sub_drone_path) - 1):
                        if self.transportation_array[sub_drone_path[i + 1]] == 4:
                            d_cost += self.Dcost[sub_drone_path[i]][sub_drone_path[i + 1]] * 2
                            sub_drone_path[i + 1] = sub_drone_path[i]
                            my_trans_matrix[i + 1] = my_trans_matrix[i]
                        else:
                            d_cost += self.Dcost[sub_drone_path[i]][sub_drone_path[i + 1]]
                    total_cost += max(t_cost, d_cost)
                    i = end - 1
            i += 1
        
        return total_cost
    
    def type_Matrix(self, path):
        arr = []
        for i in path:
            arr.append(self.transportation_array[i])
        return arr
    
    def fitness_measurement(self, ant_path):
        return 1 / self.calculate_path_cost(ant_path)

# Initialize population with binary strings
def initialize_population(population_size, chromosome_length):
    return [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(population_size)]

# Parent Selection: Roulette Wheel Selection
def select_parents(population):
    total_fitness = sum(chromosome[1] for chromosome in population)
    probabilities = [chromosome[1] / total_fitness for chromosome in population]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents
  
# Crossover: Single-point Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation: Single-bit Mutation
def mutate(chromosome, mutation_rate, max_mutations):
    mutated_chromosome = chromosome[0]  
    for _ in range(max_mutations):
        i = random.randint(0, len(chromosome) - 1)
        if random.random() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome

def create_transportation(chromosome):
    transportation_array = np.zeros_like(chromosome)
    transportation_array[0] = 3
    p = [1 / 3, 1 / 3, 1 / 3]
    for i in range(1, len(chromosome)):
        if chromosome[i] == 0:
            transportation_array[i] = np.random.choice((1, 3), size=1)
        else:
            x = np.random.choice((2, 3, 4), size=1, p = p)
            transportation_array[i] = x
            if x == 2:
                p[0] = 0
                p[1] = 1
                p[2] = 0
            elif x == 3:
                p[0] = (1 / 3) + 0.0002
                p[1] = (1 / 3) - 0.0006
                p[2] = (1 / 3) + 0.0004
            else:
                p[0] = (1 / 2)
                p[1] = (1 / 2)
                p[2] = 0

    return transportation_array
    
def GA_AS_Algo(depot_point, Ts, Ds, alpha, beta, decay, distance_matrix, population, maxitr, crossovers, mutation_rate, max_mutations):
    result = []
    pop = population[:]
    for _ in range(maxitr):
        for gpop in range(len(pop)):
            chromosome = pop[gpop]
            transportation_array = create_transportation(chromosome)
            as_al = AS_Algorithm(chromosome, distance_matrix, transportation_array, depot_point, 5, decay, alpha, beta, Ts, Ds)
            route, fitness, type_trans = as_al.run(10)
            pop[gpop] = (chromosome, fitness, route)
            if not(result) or (result[2] < fitness):
                result = (chromosome, route, fitness, type_trans, transportation_array)
        
        # Parent Selection
        mating_pool = []
        for _ in range(crossovers):
            parents = select_parents(pop)
            parent1, parent2 = parents[0], parents[1]
            mating_pool.append((parent1, parent2))
            
        # Crossover and Mutation
        offspring_population = []
        for parent1, parent2 in mating_pool:
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, max_mutations)
            child2 = mutate(child2, mutation_rate, max_mutations)
            offspring_population.extend([child1, child2])
        
        # Update population with offspring
        pop = offspring_population
    
    return result

def draw_route(path, type_transport):
    num_cities = len(path) - 1
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(num_cities):
        G.add_node(path[i])
    
    # Add edges to the graph based on transportation types
    for i in range(num_cities):
        node_i, node_next = type_transport[i], type_transport[i + 1]
        start, end = None, None
        if node_next == 1:
            if node_i == 1 or node_i == 3:
                G.add_edge(path[i], path[i+1], color='blue', arrowsize=40)  # Truck route
            elif node_i == 2:
                start = next((j for j in reversed(range(i+1)) if type_transport[j] == 3), None)
                if start:
                    G.add_edge(path[start], path[i+1], color='blue', arrowsize=40)  # Truck route
        elif node_next == 2:
            start = next((j for j in reversed(range(i+1)) if type_transport[j] == 3), None)
            end = next((k for k in range(i+1, len(type_transport)) if type_transport[k] == 3), None)
            if start != None and end != None:
                G.add_edge(path[start], path[i + 1], color='green', arrowsize=40, style='dashed')  # Drone route
                G.add_edge(path[i + 1], path[end], color='green', arrowsize=40, style='dashed')  # Drone route
            if node_i == 2:
                G.add_edge(path[i], path[i+1], color='green', arrowsize=40, style='dashed')  # Drone route
        elif node_next == 3:
            if node_i == 1 or node_i == 3:
                G.add_edge(path[i], path[i+1], color='blue', arrowsize=40)  # Truck route
            elif node_i == 2:
                start = next((j for j in reversed(range(i+1)) if type_transport[j] == 3), None)
                if start != None:
                    G.add_edge(path[start], path[i+1], color='blue', arrowsize=40)  # Truck route
        else:
            G.add_edge(path[i], path[i+1], color='red', arrowsize=40, style='dashed')  # Drone route
            G.add_edge(path[i+1], path[i], color='red', arrowsize=40, style='dashed')  # Drone route
            if type_transport[i + 2] == 3 or type_transport[i + 2] == 1:
                G.add_edge(path[i], path[i+2], color='blue', arrowsize=40)  # Truck route
            else:
                G.add_edge(path[i], path[i + 2], color='green', arrowsize=40, style='dashed')  # Drone route  
                
    pos = nx.circular_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=400)
    
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u, v in edges]
    edge_styles = ['dashed' if G[u][v]['color'] == 'green' else 'solid' for u, v in edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, style=edge_styles, arrows=True)
    
    nx.draw_networkx_labels(G, pos)
    
    depot_node = path[0]
    nx.draw_networkx_nodes(G, pos, nodelist=[depot_node], node_size=400, node_color='red')
    
    plt.title('Route')
    plt.show()

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    drone_speed= float(lines[1])  # Speed of the Drone
    truck_speed= float(lines[3])  # Speed of the Truck
    num_nodes = int(lines[5])       # Number of Nodes
    
    # Extracting coordinates and names of locations
    locations = {}
    for i in range(num_nodes):
        line = lines[i+7]
        parts = line.split()
        x_coor, y_coor = float(parts[0]), float(parts[1])
        locations[i] = (x_coor, y_coor)
    
    # Creating distance matrix
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i, (name1, (x1, y1)) in enumerate(locations.items()):
        for j, (name2, (x2, y2)) in enumerate(locations.items()):
            distance_matrix[i][j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return truck_speed, drone_speed, distance_matrix

def stability_test(num_runs, depot_point, Ts, Ds, alpha, beta, decay, distance_matrix, population, maxitr, crossovers, mutation_rate, max_mutations):
    results = []
    for i in range(num_runs):
        _, path, fitness,  type_transport, transportation_array = GA_AS_Algo(depot_point, Ts, Ds, alpha, beta, decay, distance_matrix, population, maxitr, crossovers, mutation_rate, max_mutations)
        results.append((1 / fitness, path, type_transport, transportation_array))
        print("---------------------")
        print("Test ", i + 1)
        print("Path: ", path)
        print("Fitness: %.5f - Cost: %.2f" % (fitness, 1/fitness))
        print("Type Transport:", type_transport)
    return results

def plot_distribution(results):
    fitness_values = [result[0] for result in results]
    plt.hist(fitness_values, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fitness Values')
    plt.show()
  
def plot_stability_test(results):
    fits_values = [1 / result[0] for result in results]
    differences = [max(fits_values) - value  for value in fits_values]
    plt.plot(range(1, len(results) + 1), differences, marker='o', color='blue', linestyle='-')
    plt.xlabel('Run')
    plt.ylabel('Difference from Best Fitness')
    plt.title('Stability Test')
    plt.grid(True)
    plt.show()

def measure_execution_time(file_path, depot_point, alpha, beta, decay, maxitr, crossovers, mutation_rate, max_mutations):
    Ts, Ds, distance_matrix = read_input_file(file_path)
    population_size = len(distance_matrix)
    population = initialize_population(population_size, len(distance_matrix))
    
    start_time = time.time()
    result = GA_AS_Algo(depot_point, Ts, Ds, alpha, beta, decay, distance_matrix, population, maxitr, crossovers, mutation_rate, max_mutations)
    end_time = time.time()
    
    execution_time = end_time - start_time
    return execution_time, result

def compare_execution_times(file_paths, depot_point, alpha, beta, decay, maxitr, crossovers, mutation_rate, max_mutations):
    execution_times = []
    results = []
    
    for file_path in file_paths:
        exec_time, result = measure_execution_time(file_path, depot_point, alpha, beta, decay, maxitr, crossovers, mutation_rate, max_mutations)
        execution_times.append((file_path, exec_time))
        results.append(result)
    
    return execution_times, results

def plot_execution_times(execution_times):
    file_names = []
    
    for file_path, _ in execution_times:
        s = file_path.rfind('n')
        e = file_path.rfind('.')
        file_names.append(file_path[s:e])
    exec_times = [exec_time for _, exec_time in execution_times]

    plt.figure(figsize=(10, 6))
    plt.bar(file_names, exec_times, color='skyblue')
    plt.xlabel('Số lượng node')
    plt.ylabel('Thời gian thực thi(s)')
    plt.title('Biểu đồ thời gian thực thi theo số lượng node')
    # plt.grid(True)
    plt.show()
    
file_paths = [
    "tsp_instances/uniform-1-n5.txt",
    "tsp_instances/uniform-1-n10.txt",
    "tsp_instances/uniform-1-n12.txt",
    "tsp_instances/uniform-1-n14.txt",
    "tsp_instances/uniform-1-n16.txt",
    "tsp_instances/uniform-1-n18.txt",
    "tsp_instances/uniform-1-n20.txt",
    "tsp_instances/uniform-1-n50.txt",
    "tsp_instances/uniform-1-n75.txt",
]    

depot_point = 0

file_path = "tsp_instances/uniform-1-n10.txt"
Ts, Ds, distance_matrix = read_input_file(file_path)

# α and β are pre-given parameters regulating how much the pheromone amount and distance will influence the selection of the next node in the calculation.
alpha = 1  # GA-AS constant
beta = 1  # GA-AS constant
decay = 0.5  # GA-AS constant

# Số vòng lặp tối đa
maxitr = 5
crossovers = 5
max_mutations = 5
mutation_rate = 0.1

population = initialize_population(len(distance_matrix), len(distance_matrix))

# Chạy thử hàm stability_test và vẽ biểu đồ phân phối của chi phí lộ trình
num_runs = 5
results = stability_test(num_runs, depot_point, Ts, Ds, alpha, beta, decay, distance_matrix, population, maxitr, crossovers, mutation_rate, max_mutations)

print(results)
draw_route(results[0][1], results[0][2])

# plot_distribution(results)
# plot_stability_test(results)

# execution_times, results = compare_execution_times(file_paths, depot_point, alpha, beta, decay, maxitr, crossovers, mutation_rate, max_mutations)

# for file_path, exec_time in execution_times:
    # print(f"Execution time for {file_path}: {exec_time:.2f} seconds")

# plot_execution_times(execution_times)







