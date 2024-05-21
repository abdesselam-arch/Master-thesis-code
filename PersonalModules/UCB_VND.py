import math
import random

from PersonalModules.utilities import dijkstra, epsilon_constraints, get_Diameter, len_free_slots, len_sinked_relays, monitor_performance, plot_fitness_improvement, plot_histogram, print_sequence_counts, track_neighborhood_sequence, write_sequence_counts_to_json
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
        sinked_relays.append((chosen_random_slot, hop_count(sink, chosen_random_slot, sentinels, mesh_size)))
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
                sinked_relays.append((chosen_slot, hop_count(sink, chosen_slot, sentinels, mesh_size)))
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

def hop_count(sink, relay, sentinels, mesh_size):
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
    return int(hop_count /mesh_size)

@monitor_performance
# UCB1 - Variable Neighborhood Descent ---------------------------------------------------------------------------
def UCB_VND(grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, lmax, alpha, beta):
    l = 1  # Neighborhood counter
    qualities = [0.0] * lmax
    iteration = 0
    n_free_slots, n_sinked_relays = [], []
    neighborhood_counts = [0] * lmax
    #max_iterations = len_sinked_relays(sinked_relays)
    max_iterations = int(grid /mesh_size)**2
    # max_iterations = 1000
    print(f'Max num of iterations: {max_iterations}')
    total_number_actions = 0
    exploration_factor = 2
    consecutive_errors = 0
    
    # For the termination criteria
    velocities = []
    velocity = 0
    velocity_length = 6
    avg_velocity = 0
    patience = 2

    fitness_values = []
    iteration_numbers = []
    sequence_counts = {}
    prior_neighborhood = None  # Initialize prior neighborhood to None
    neighborhood_iteration = []

    # The near optimal solution to be returned at the end
    optimal_sinked_relays, optimal_free_slots = None, None
    best_solution_relays = float('inf')
    
    previous = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, 0, mesh_size, alpha, beta)
    for l in range(lmax):
        qualities[l] += previous

    # Shaking operation GVNS
    n_free_slots, n_sinked_relays, action, remember_used_relays = shaking(sinked_sentinels, sinked_relays, free_slots, custom_range, sink, mesh_size)
    # while (iteration <= max_iterations) and (consecutive_errors <= 6):
    while consecutive_errors <= 1:
        i = 0  # Neighbor counter
        improvement = True  # Flag to indicate improvement
        previous = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, 0, mesh_size, alpha, beta)
        
        l, neighborhood_count = Upper_Confidence_Bound.UCB1_policy(lmax, qualities, exploration_factor, total_number_actions)
        neighborhood_counts[l] += 1

        # Update sequence counts
        track_neighborhood_sequence(prior_neighborhood, l, sequence_counts)
        while improvement and i < len(sinked_relays) + 1:
            improvement = False
            i += 1
            prior_neighborhood = l  # Update prior neighborhood
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
            
            after = epsilon_constraints(grid, n_free_slots, sink, n_sinked_relays, sinked_sentinels, 0, mesh_size, alpha, beta)
            total_number_actions += 1

            distance_bman, sentinel_bman, cal_bman = dijkstra(grid, sink, sinked_relays, sinked_sentinels)
            if (after < previous) and (sentinel_bman.count(999) == 0):
                print(f'\nPrevious Fitness: {previous}, After Fitness: {after}')
                free_slots, sinked_relays = n_free_slots.copy(), n_sinked_relays.copy()
                improvement = True
                consecutive_errors = 0
                # Update the qualities[] of the chosen neighborhoods - Rewarding the best Action
                qualities[l] += Upper_Confidence_Bound.Credit_Assignment(improvement, previous, after, l) / neighborhood_count 
                velocity = abs(previous - after)
                previous = after
                fitness_values.append(after)
                neighborhood_iteration.append(l + 1)
                iteration_numbers.append(iteration)

                velocities.append(velocity)
                last_n_velocities = velocities[-velocity_length:]
                print(f'\n\n\nLast {velocity_length} velocities: {last_n_velocities}')

                if after < best_solution_relays:
                    best_solution_relays = after
                    optimal_sinked_relays = sinked_relays.copy() 
                    optimal_free_slots = free_slots.copy()
                    
                    print(f'\nThe current optimal solution: {len_sinked_relays(optimal_sinked_relays)} relays deployed')
                    print(f'The current optimal solution: {len_free_slots(grid, optimal_sinked_relays)} free slots remaining\n\n')
            else:
                iteration += 1
                print(f'\n\nReinforcement Learning Episode: {get_ordinal_number(iteration)}')
                last_n_velocities = velocities[-velocity_length:]
                avg_velocity = (sum(last_n_velocities) or 0) / (len(last_n_velocities) or 1)
                print(f'The AVG velocity: {avg_velocity}')
                if (max_iterations - iteration <= int(grid/mesh_size)):
                    exploration_factor = 0
                if (iteration >= max_iterations) and (avg_velocity <= 1):
                    consecutive_errors += 1 
                    print(f'\n\n     (LS) Consecutive errors: {consecutive_errors}')

        # else:
        # Update the qualities[] of the chosen neighborhoods - Regret the best Action
        qualities[l] += Upper_Confidence_Bound.Credit_Assignment(improvement, previous, after, l)
        print(f'\n {l+1} Neighborhood Previous Fitness: {previous}, After Fitness: {after}')
        print(f'\nThere are {len_sinked_relays(sinked_relays)} relays deployed')
        print(f'There are {len_free_slots(grid, sinked_relays)} free slots remaining\n\n')
        
    
    #plot the histogram    
    # (toggle it when we want) plot_histogram(neighborhood_counts, max_iterations) 
    plot_fitness_improvement(iteration_numbers, fitness_values, neighborhood_iteration)
    # print_sequence_counts(sequence_counts)
    '''write_sequence_counts_to_json(grid_size=int(grid/mesh_size), sink_coordinates= sink, num_relays_deployed = len_sinked_relays(sinked_relays),
                                  diameter=get_Diameter(sentinel_bman, cal_bman, mesh_size), sequence_counts = sequence_counts, fitness= best_solution_relays,
                                  filename= f'{int(grid/mesh_size)}x{int(grid/mesh_size)} sequence counts', filepath = "C:/Users/nouri/OneDrive/Desktop/Papers/Python program files/Python program files/Neighborhoods sequence")'''
    print(f'neighborhood at every iteration: {neighborhood_iteration}')

    return optimal_sinked_relays, optimal_free_slots