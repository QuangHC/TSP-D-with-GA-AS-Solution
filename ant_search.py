import numpy as np
import random

# Mảng gồm 10 phần tử, với 0 là phương tiện Drone, 1 là phương tiện Truck.
population = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

# Mảng chứa loại phương tiện phục vụ cho từng khách hàng 
# (1: Truck, 2: Drone (bay thẳng), 3: Khả năng vận chuyển cả 2 loại, 4: Drone (quay lại)).
transportation_array = [3, 3, 3, 4, 2, 1, 3, 3, 3, 2]

# Chỉ số khách hàng trung tâm (depot)
depot_point = 0

# Tập hợp chứa các chỉ mục khách hàng
tsp_set = set(range(len(population)))

# Hằng số
Ts = 10  # Speed of Truck
Ds = 20  # Speed of Drone
alpha = 1  # GA-AS constant
beta = 1  # GA-AS constant
decay = 0.95  # GA-AS constant

# Created distance matrix for 10 customers
distance_matrix = np.array([
    [0, 29, 20, 21, 16, 31, 100, 12, 4, 31],
    [29, 0, 15, 29, 28, 40, 72, 21, 29, 41],
    [20, 15, 0, 15, 14, 25, 81, 9, 23, 27],
    [21, 29, 15, 0, 4, 12, 92, 12, 25, 13],
    [16, 28, 14, 4, 0, 16, 94, 9, 20, 16],
    [31, 40, 25, 12, 16, 0, 95, 24, 36, 3],
    [100, 72, 81, 92, 94, 95, 0, 90, 101, 99],
    [12, 21, 9, 12, 9, 24, 90, 0, 15, 25],
    [4, 29, 23, 25, 20, 36, 101, 15, 0, 35],
    [31, 41, 27, 13, 16, 3, 99, 25, 35, 0]
])

class AS_Algorithm:
    def __init__(self, chromosome, distances, n_ants, decay, alpha, beta, Ts, Ds) -> None:
        self.chromosome = chromosome
        self.truck = [i for i in range(len(chromosome)) if chromosome[i] == 0]
        self.drone = [i for i in range(len(chromosome)) if chromosome[i] == 1]
        self.distances = distances
        self.truckphmtrix = np.ones_like(distances) / len(distances) 
        self.dronephmtrix = np.ones_like(distances) / len(distances)
        self.all_inds = range(len(distances)) 
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
        return best_path, all_time_best_cost

    def generate_ants(self):
        ants = []
        for _ in range(self.n_ants):
            ant_path = self.generate_ant_path(depot_point)
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
            if transportation_array[ant_path[-1]] == 1:
                new_unvisited = list(
                    filter(lambda x: transportation_array[x] == 1 or transportation_array[x] == 3, unvisited))
            elif transportation_array[ant_path[-1]] == 2:
                new_unvisited = list(filter(lambda x: transportation_array[x] == 3, unvisited))
            elif transportation_array[ant_path[-1]] == 3:
                new_unvisited = list(filter(lambda x: transportation_array[x] != 1, unvisited))
            else:
                new_unvisited = list(
                    filter(lambda x: transportation_array[x] == 3 or transportation_array[x] == 2, unvisited))
            next_city = self.choose_next_city(ant_path[-1], new_unvisited if len(new_unvisited) else unvisited)
            if next_city is None:
                return [], 10**5
            ant_path.append(next_city)
            unvisited.remove(next_city)
            
        return ant_path

    def choose_next_city(self, current_city, unvisited):
        pheromone_values = np.zeros_like(distance_matrix, dtype=float)
        for j in unvisited:
            if j in self.truck:
                p_tij = self.truckphmtrix[current_city][j]
                d_ij = self.distances[current_city][j]
                pheromone_values[current_city][j] = (p_tij ** alpha) * ((1 / d_ij) ** beta)
            else:
                p_dij = self.dronephmtrix[current_city][j]
                d_ij = self.distances[current_city][j]
                pheromone_values[current_city][j] = (p_dij ** alpha) * ((1 / d_ij) ** beta)

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
        self.truckphmtrix = (1 - decay) * self.truckphmtrix + pheromone_truck_change
        self.dronephmtrix = (1 - decay) * self.dronephmtrix + pheromone_drone_change

    def get_best_path(self, ants):
        best_path = None
        best_cost = float('inf')
        for ant_path, cost in ants:
            if cost < best_cost:
                best_path = ant_path
                best_cost = cost
        return best_path, best_cost

    def calculate_path_cost(self, path):
        total_cost = 0
        for i in range(len(path) - 1):
            if transportation_array[path[i + 1]] == 3 or transportation_array[path[i + 1]] == 1:
                total_cost += self.Tcost[path[i]][path[i + 1]]
            elif transportation_array[path[i + 1]] == 4:
                total_cost += self.Dcost[path[i]][path[i + 1]] * 2
            else:
                total_cost += max(self.Tcost[path[i]][path[i + 1]], self.Dcost[path[i]][path[i + 1]])
        return total_cost
    
    def type_Matrix(self, path):
        arr = []
        for i in path:
            arr.append(transportation_array[i])
        return arr
    
    def fitness_measurement(self, ant_path):
        return 1 / self.calculate_path_cost(ant_path)
    

population = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

al = AS_Algorithm(population, distance_matrix, 5, decay, alpha, beta, Ts, Ds)
path, cost = al.run(100)
print(path, cost, al.fitness_measurement(path))
    
