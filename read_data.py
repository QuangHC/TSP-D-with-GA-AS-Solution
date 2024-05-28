import numpy as np

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    truck_speed = float(lines[1])  # Speed of the Truck
    drone_speed = float(lines[3])  # Speed of the Drone
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

# Path to the input file
file_path = "tsp_instances/uniform-1-n5.txt"

# Read input file and assign values
truck_speed, drone_speed, distance_matrix = read_input_file(file_path)

print("Truck Speed:", truck_speed)
print("Drone Speed:", drone_speed)
print("Distance Matrix:\n", distance_matrix)
