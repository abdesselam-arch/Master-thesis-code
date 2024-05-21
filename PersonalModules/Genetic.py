import itertools
import random
import math

from matplotlib import pyplot as plt

from PersonalModules.utilities import dijkstra

def crossover(parent1, parent2):
    # Perform crossover between two parents using Uniform Crossover

    # Choose the solution with fewer nodes as the primary parent
    if sum(len(route) for route in parent1) < sum(len(route) for route in parent2):
        primary_parent = parent1
        secondary_parent = parent2
    else:
        primary_parent = parent2
        secondary_parent = parent1
    
    child = []
    for gene1, gene2 in zip(primary_parent, secondary_parent):
        # Randomly select the gene from either parent with a 50% probability
        if random.random() < 0.5:
            child.append(gene1)
        else:
            child.append(gene2)
    print('Success! Crossover operation complete')
    return child

def mutate(solution, free_slots, custom_range):
    # Perform mutation on the solution
    mutated_solution = solution.copy()
    sentinel_index = random.randint(0, len(mutated_solution) - 1)
    current_node = mutated_solution[sentinel_index][-1]

    # Check if free_slots is empty
    if not (free_slots):
        print("No mutation!")
        return mutated_solution

    # Randomly select a slot
    chosen_node = random.choice(free_slots)
    if chosen_node in mutated_solution[sentinel_index]:
        # Remove relay if the slot is already occupied
        mutated_solution[sentinel_index].remove(chosen_node)
        free_slots.append(chosen_node)  # Add the slot back to free slots
    else:
        # Add relay if the slot is free
        mutated_solution[sentinel_index].append(chosen_node)
        free_slots.remove(chosen_node)  # Remove the slot from free slots
    current_node = chosen_node
    if current_node in mutated_solution[sentinel_index][:-1]:
        # If the chosen slot is in the sentinels, choose again
        nearby_candidates = []
    else:
        nearby_candidates = [node for node in free_slots if math.dist(current_node, node) < custom_range]

    print('Success! Mutation operation complete')
    return mutated_solution

def evaluate(solution, sink, sinked_relays, grid, free_slots, sinked_sentinels, mesh_size):
    #extract sinked relays
    sinked_sentinels = [route[0] for route in solution]
    sinked_relays = [relay for route in solution for relay in route[1:]]
    
    # Evaluate the fitness of the solution
    # Two objectives diameter, number of relays
    distance_bman, sentinel_bman, cal_bman = dijkstra(grid, sink, sinked_relays, sinked_sentinels)
    fitness = (0.3 * len(sinked_relays)) + (0.3 * (cal_bman / mesh_size))
    if 999 in sentinel_bman:
        return fitness + 999
    else:      
        return fitness

def calculate_min_hop_count(sink, sinked_relays, mesh_size):
    min_hop_counts = []
    for relay in sinked_relays:
        # Calculate Manhattan distance (hop count) from relay to the sink
        distance = abs(sink[0] - relay[0]) + abs(sink[1] - relay[1])
        distance = distance / mesh_size
        min_hop_counts.append(distance)
    return min_hop_counts

def initial_population(population_size, sinkless_sentinels, free_slots, max_hops_number, custom_range, grid, sink):
    population = []
    total_slots = len(free_slots)
    remaining_slots = total_slots

    for _ in range(population_size):
        sentinel_solution = []
        used_slots = set()

        for sentinel in sinkless_sentinels:
            route = [sentinel]
            current_node = sentinel

            # Track the number of relays connected to the sink
            relays_connected_to_sink = 0

            while current_node != sink and relays_connected_to_sink < max_hops_number:
                nearby_candidates = [node for node in free_slots if math.dist(current_node, node) < custom_range]

                if not nearby_candidates:
                    # If no candidates are available, break the loop
                    break

                # Choose the nearest candidate
                nearest_candidate = min(nearby_candidates, key=lambda node: math.dist(node, sink))

                # Update current node and remove chosen candidate from free slots
                current_node = nearest_candidate
                free_slots.remove(nearest_candidate)
                used_slots.add(nearest_candidate)

                route.append(nearest_candidate)

                # If the nearest candidate is a relay, increment the count
                if nearest_candidate != sink:
                    relays_connected_to_sink += 1

            # Add the route to the solution for the current sentinel
            sentinel_solution.append(route)

        # Add any unused slots to random routes
        unused_slots = list(free_slots)
        random.shuffle(unused_slots)
        for slot in unused_slots:
            route_index = random.randint(0, len(sentinel_solution) - 1)
            sentinel_solution[route_index].append(slot)
            used_slots.add(slot)

        # Update remaining slots
        remaining_slots -= len(used_slots)

        # Randomly delete 10 nodes excluding nodes on the diagonals
        diagonal_nodes = set()
        for i in range(len(free_slots)):
            for j in range(i + 1, len(free_slots)):
                if abs(free_slots[i][0] - free_slots[j][0]) == abs(free_slots[i][1] - free_slots[j][1]):
                    diagonal_nodes.add(free_slots[i])
                    diagonal_nodes.add(free_slots[j])

        nodes_to_delete = random.sample(used_slots - diagonal_nodes, min(int(grid / 20), len(used_slots) - len(diagonal_nodes)))
        for node in nodes_to_delete:
            for route in sentinel_solution:
                if node in route:
                    route.remove(node)
                    used_slots.remove(node)
                    break

        # Add the solution for the current population member
        population.append(sentinel_solution)

    print(f"Total slots: {total_slots}, Slots used: {total_slots - remaining_slots}, Slots remaining: {remaining_slots}")

    return population

def genetic_algorithm(population_size, generations, sink, sinkless_sentinels, free_slots, max_hops_number, custom_range, mesh_size):

    First_time = True
    sentinels = sinkless_sentinels[:]
    grid = len(free_slots) + len(sinkless_sentinels) + 1
    print("The grid =", grid)
    sinked_sentinels, sinked_relays, sinkless_relays, candidate_slots = [], [], [], []
    found_forbidden, ERROR = False, False

    fitness_per_generation, all_fitness_scores = [], []

    population = initial_population(population_size, sinkless_sentinels, free_slots, max_hops_number, custom_range, grid, sink)

    for generation in range(generations):
        print(f'Generation {generation+1}')

        # Evaluate the fitness of each solution in the population
        fitness_scores = [evaluate(solution, sink, sinked_relays, grid, free_slots, sinked_sentinels, mesh_size) for solution in population]
        all_fitness_scores.append(fitness_scores)  # Store fitness scores of all solutions

        # Select parents based on their fitness
        parents_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[:2]
        parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]

        # Perform crossover and mutation to generate a new solution
        child = crossover(parent1, parent2)
        child = mutate(parent1, free_slots, custom_range)
        child = parent1

        # Append the best fitness score of this generation to fitness_per_generation
        fitness_per_generation.append(max(fitness_scores))

        # Replace the least fit solution in the population with the new child
        min_fitness_index = fitness_scores.index(max(fitness_scores))
        population[min_fitness_index] = child

        all_fitness_scores= []

    # Evaluate the final population and select the best solution
    fitness_scores = [evaluate(solution, sink, sinked_relays, grid, free_slots, sinked_sentinels, mesh_size) for solution in population]
    best_solution_index = fitness_scores.index(max(fitness_scores))
    best_solution = population[best_solution_index]

    # Extract relevant variables to match the outputs of greedy_algorithm
    sinked_sentinels = [route[0] for route in best_solution]
    sinked_relays = [relay for route in best_solution for relay in route[1:]]
    free_slots_remaining = [slot for slot in free_slots if slot not in sum(best_solution, [])]
   
    min_hop_counts = calculate_min_hop_count(sink, sinked_relays, mesh_size)
    sinked_relays = list(zip(sinked_relays, min_hop_counts))

    print('\nSinked Sentinels\n',sinked_sentinels)
    print('\nSinked Relays\n', sinked_relays)

    Finished = True  # Placeholder, update based on your termination criteria
    ERROR = False  # Placeholder, update based on error conditions

    return sinked_sentinels, sinked_relays, free_slots_remaining, Finished, ERROR
