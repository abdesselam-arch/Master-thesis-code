import math
import random

from matplotlib import pyplot as plt
import numpy as np

from PersonalModules.utilities import bellman_ford, bellman_ford_parallel, epsilon_constraints, len_free_slots, len_sinked_relays
from PersonalModules import Upper_Confidence_Bound
from main import get_ordinal_number

'''
    5 Neighborhoods
'''

def add_next_to_relay(sentinels, sink, sinked_relays, free_slots, remember_used_relays, custom_range, mesh_size):
    performed_action = []
    candidate_slots = []
    
    allowed_sinked_relays = [x for x in sinked_relays if x not in remember_used_relays]
    min_meshes, max_meshes = get_min_max_meshes(allowed_sinked_relays, free_slots, custom_range)
    chosen_random_relay = max_meshes[0][0]
    
    for i in range(len(free_slots)):
        if math.dist(chosen_random_relay, free_slots[i]) < custom_range:
            candidate_slots.append(free_slots[i])
            
    if candidate_slots:
        chosen_random_slot = random.choice(candidate_slots)
        # sinked_relays.append((chosen_random_slot, (abs(sink[0] - chosen_random_slot[0]) + abs(sink[1] - chosen_random_slot[1])) / mesh_size))
        # sinked_relays.append((chosen_random_slot, 1))
        sinked_relays.append((chosen_random_slot, hop_count(sink, chosen_random_slot, sentinels)))
        performed_action = [1, chosen_random_slot, "(P) Add a relay next to a random connected relay's neighborhood."]
        free_slots.remove(chosen_random_slot)
        remember_used_relays.append(chosen_random_slot)
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def delete_random_relay(sentinels, sinked_relays, free_slots, remember_used_relays, custom_range):
    performed_action = []
    min_meshes, max_meshes = get_min_max_meshes(sinked_relays, free_slots, custom_range)
    chosen_random_relay = min_meshes[0][0]
    
    if sinked_relays:
        sinked_relays = [relay for relay in sinked_relays if relay[0] != chosen_random_relay]
        performed_action = [2, chosen_random_relay, "(P) Delete a random connected relay."]
        free_slots.append(chosen_random_relay)
        remember_used_relays.append(chosen_random_relay)
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def delete_relay_next_to_sentinel(sentinels, sinked_relays, free_slots, remember_used_relays, custom_range):
    performed_action = []
    
    # Dictionary to store the count of relays next to each sentinel
    sentinel_neighbor_count = {tuple(sentinel): 0 for sentinel in sentinels}
    
    # Count the number of relays next to each sentinel
    for sentinel in sentinels:
        for relay in sinked_relays:
            if math.dist(sentinel, relay[0]) < custom_range:
                sentinel_neighbor_count[tuple(sentinel)] += 1
    
    # Filter out sentinels with multiple relay neighbors
    sentinels_with_multiple_neighbors = [sentinel for sentinel, count in sentinel_neighbor_count.items() if count > 1]
    
    # Select a random sentinel with multiple relay neighbors, if any
    if sentinels_with_multiple_neighbors:
        chosen_sentinel = random.choice(sentinels_with_multiple_neighbors)
        
        # Find relays next to the chosen sentinel
        relays_next_to_chosen_sentinel = [relay for relay in sinked_relays if math.dist(chosen_sentinel, relay[0]) < custom_range]
        
        # Delete a random relay next to the chosen sentinel
        if relays_next_to_chosen_sentinel:
            chosen_random_relay = random.choice(relays_next_to_chosen_sentinel)
            sinked_relays = [relay for relay in sinked_relays if relay[0] != chosen_random_relay[0]]
            performed_action = [3, chosen_random_relay[0], "(P) Delete a random relay that's next to a sentinel with multiple relay neighbors"]
            free_slots.append(chosen_random_relay[0])
            remember_used_relays.append(chosen_random_relay[0])
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def relocate_relay(sinked_sentinels, sink, sinked_relays, free_slots, remember_used_relays, custom_range, mesh_size):
    performed_action = []
    
    if sinked_relays and free_slots:
        # Choose a relay and a free slot randomly
        '''relay_index = random.randint(0, len(sinked_relays) - 1)
        free_slot_index = random.randint(0, len(free_slots) - 1)
        
        # Swap the positions of the relay and the free slot
        relay_position = sinked_relays[relay_index][0]
        free_slot_position = free_slots[free_slot_index]
        sinked_relays[relay_index] = (free_slot_position, sinked_relays[relay_index][1])
        free_slots[free_slot_index] = relay_position
        sinked_relays = [relay for relay in sinked_relays if relay[0] != relay_position]

        # Update the performed action to reflect the swap
        performed_action = [2, "(LS) Swap relay with free slot"]'''
        free_slots, sinked_relays, action, remember_used_relays = delete_random_relay(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
        free_slots, sinked_relays, action, remember_used_relays = add_next_to_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
        # Update the performed action to reflect the swap
        performed_action = [2, "(LS) Swap relay with free slot"]
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def add_relay_next_to_sentinel(sentinels, sink, sinked_relays, free_slots, remember_used_relays, custom_range, mesh_size):
    performed_action = []
    
    for sentinel in sentinels:
        no_neighbors = True
        for relay in sinked_relays:
            if math.dist(sentinel, relay[0]) < custom_range:
                no_neighbors = False
                break
        
        if no_neighbors:
            candidate_slots = [slot for slot in free_slots if math.dist(sentinel, slot) < custom_range]
            if candidate_slots:
                chosen_slot = random.choice(candidate_slots)
                # sinked_relays.append((chosen_slot, (abs(sink[0] - chosen_slot[0]) + abs(sink[1] - chosen_slot[1])) / mesh_size))
                # sinked_relays.append((chosen_slot, 1))
                sinked_relays.append((chosen_slot, hop_count(sink, chosen_slot, sentinels)))
                performed_action = [1, chosen_slot, "(P) Add a relay next to a sentinel with no relay neighbors."]
                free_slots.remove(chosen_slot)
                remember_used_relays.append(chosen_slot)
                break  # Only add one relay per iteration
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

'''
    Helper functions
'''

def get_min_max_meshes(sinked_relays, free_slots, custom_range):
    min_meshes_candidate = []
    max_meshes_candidate = []
    random.shuffle(sinked_relays)

    for i in range(len(sinked_relays)):
        empty_meshes_counter = 0

        # Calculate meshes around a sinked relay
        for j in range(len(free_slots)):
            if math.dist(sinked_relays[i][0], free_slots[j]) < custom_range:
                empty_meshes_counter = empty_meshes_counter + 1

        if len(min_meshes_candidate) != 0 and len(max_meshes_candidate) != 0:

            # Acquire minimum meshes
            if min_meshes_candidate[1] > empty_meshes_counter:
                min_meshes_candidate = [sinked_relays[i], empty_meshes_counter]

            # Acquire maximum meshes
            if max_meshes_candidate[1] < empty_meshes_counter:
                max_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
        else:
            min_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
            max_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
    return min_meshes_candidate, max_meshes_candidate

def shaking(sinked_sentinels, sinked_relays, free_slots, custom_range, sink, mesh_size):
    for shaking_neighborhood in range(1,5):
        if shaking_neighborhood == 1:
            # 1st neighborhood
            free_slots, sinked_relays, action, remember_used_relays = add_next_to_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
        if shaking_neighborhood == 2:
            # 2nd neighborhood
            free_slots, sinked_relays, action, remember_used_relays = delete_random_relay(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
        if shaking_neighborhood == 3:
            # 3rd neighborhood
            free_slots, sinked_relays, action, remember_used_relays = delete_relay_next_to_sentinel(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
        if shaking_neighborhood == 4:
            # 4th neighborhood
            free_slots, sinked_relays, action, remember_used_relays = relocate_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
        if shaking_neighborhood == 5:
            # 5th neighborhood
            free_slots, sinked_relays, action, remember_used_relays = add_relay_next_to_sentinel(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
    print('Shaking operation done!')
    return free_slots, sinked_relays, action, remember_used_relays

def hop_count(sink, relay, sentinels):
    current_position = relay
    hop_count = 0
    # Iterate until the current position reaches the sink
    while current_position != sink:
        # Find neighbors of the current position
        neighbors = [(current_position[0] + 1, current_position[1]),
                     (current_position[0] - 1, current_position[1]),
                     (current_position[0], current_position[1] + 1),
                     (current_position[0], current_position[1] - 1)]
        neighbors = [neighbor for neighbor in neighbors if neighbor not in [position for position in sentinels]]
        # Calculate distances to each neighbor
        distances = [abs(sink[0] - neighbor[0]) + abs(sink[1] - neighbor[1]) for neighbor in neighbors]
        # Choose the neighbor with the minimum distance to the sink
        min_index = distances.index(min(distances))
        current_position = neighbors[min_index]
        hop_count += 1
    return int(hop_count/20)

def plot_histogram(neighborhoods_selected):
    """
    Plot a histogram of neighborhoods selections.

    Args:
        neighborhoods_selected (list): List of neighborhoods selected by the algorithm.

    Returns:
        None
    """
    # Count the occurrences of each neighborhood
    neighborhood_counts = [neighborhoods_selected.count(i) for i in range(len(neighborhoods_selected))]
    plt.hist(neighborhood_counts)
    plt.title('Histogram of Neighborhoods selections')
    plt.xlabel('Neighborhoods')
    plt.ylabel('Number of times each neighborhood was selected by the algorithm')
    plt.show()

# UCB1 - Variable Neighborhood Descent ---------------------------------------------------------------------------
def UCB_VND(grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, lmax, alpha, beta):
    l = 1  # Neighborhood counter
    qualities = [0.0] * lmax
    iteration = 0
    n_free_slots, n_sinked_relays = [], []
    total_reward = 1 
    max_iterations = int(grid /20) * int(grid /20)
    print(f'Max num of iterations: {max_iterations}')
    total_number_actions = 0

    # The near optimal solution to be returned at the end
    optimal_sinked_relays, optimal_free_slots = None, None
    best_solution_relays = float('inf')
    distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
    previous = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size, alpha, beta)
    for l in range(lmax):
        qualities[l] += previous

    # Shaking operation GVNS
    n_free_slots, n_sinked_relays, action, remember_used_relays = shaking(sinked_sentinels, sinked_relays, free_slots, custom_range, sink, mesh_size)
    while iteration <= max_iterations:
        i = 0  # Neighbor counter
        improvement = True  # Flag to indicate improvement

        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
        print(f'Previous sentinel_bman: {sentinel_bman}')
        # sentinel_bman, cal_bman = bellman_ford_parallel(grid, free_slots, sink, sinked_relays, sinked_sentinels)
        previous = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size, alpha, beta)
        
        l, neighborhood_count = Upper_Confidence_Bound.UCB1_policy(lmax, qualities, 2, total_number_actions)
        while improvement and i < len(sinked_relays) + 1:
            improvement = False
            i += 1
            if l == 0:
                n_free_slots, n_sinked_relays, action, remember_used_relays = add_next_to_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
                print('Relay added')
            
            elif l == 1:
                n_free_slots, n_sinked_relays, action, remember_used_relays = delete_random_relay(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
                print('Random relay deleted ')
            
            elif l == 2:
                n_free_slots, n_sinked_relays, action, remember_used_relays = delete_relay_next_to_sentinel(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
                print('Relay next to sentinel deleted')
            
            elif l == 3:
                n_free_slots, n_sinked_relays, action, remember_used_relays = relocate_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
                print('Relays relocated')
            
            elif l == 4:
                n_free_slots, n_sinked_relays, action, remember_used_relays = add_relay_next_to_sentinel(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
                print('Relay added next to sentinel with no neighbors')
            
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
            after = epsilon_constraints(grid, n_free_slots, sink, n_sinked_relays, sinked_sentinels, cal_bman, mesh_size, alpha, beta)
            total_number_actions += 1

            if (after < previous) and (sentinel_bman.count(999) == 0):
                print(f'\nPrevious Fitness: {previous}, After Fitness: {after}')
                free_slots, sinked_relays = n_free_slots, n_sinked_relays
                improvement = True
                # Update the qualities[] of the chosen neighborhoods - Rewarding the best Action
                qualities[l] += Upper_Confidence_Bound.Credit_Assignment(improvement, previous, after, l) / neighborhood_count 
                previous = after

                if after < best_solution_relays:
                    best_solution_relays = after
                    optimal_sinked_relays = sinked_relays.copy() 
                    optimal_free_slots = free_slots.copy()
                    
                    print(f'\nThe current optimal solution: {len_sinked_relays(optimal_sinked_relays)} relays deployed')
                    print(f'The current optimal solution: {len_free_slots(grid, optimal_sinked_relays)} free slots remaining\n\n')
            else:
                iteration += 1
                print(f'\n\nReinforcement Learning Episode: {get_ordinal_number(iteration)}')

        # else:
        # Update the qualities[] of the chosen neighborhoods - Regret the best Action
        qualities[l] += Upper_Confidence_Bound.Credit_Assignment(improvement, previous, after, l)
        print(f'\n {l+1} Neighborhood Previous Fitness: {previous}, After Fitness: {after}')
        print(f'\nThere are {len_sinked_relays(sinked_relays)} relays deployed')
        print(f'There are {len_free_slots(grid, sinked_relays)} free slots remaining\n\n')

        '''distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
        print(f'UCB_VND Sentinel: {sentinel_bman}')'''

    return optimal_sinked_relays, optimal_free_slots