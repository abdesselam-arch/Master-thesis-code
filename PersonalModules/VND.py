import math
import random

import numpy as np

from PersonalModules.utilities import bellman_ford, epsilon_constraints, get_stat
from PersonalModules import Upper_Bound_Confident

'''
    5 Neighborhoods
'''

def add_next_to_relay(sentinels, sinked_relays, free_slots, remember_used_relays):
    performed_action = []
    candidate_slots = []
    
    allowed_sinked_relays = [x for x in sinked_relays if x not in remember_used_relays]
    min_meshes, max_meshes = get_min_max_meshes(allowed_sinked_relays, free_slots)
    chosen_random_relay = max_meshes[0][0]
    
    for i in range(len(free_slots)):
        if math.dist(chosen_random_relay, free_slots[i]) < 30:
            candidate_slots.append(free_slots[i])
            
    if candidate_slots:
        chosen_random_slot = random.choice(candidate_slots)
        sinked_relays.append([chosen_random_slot, 1])
        performed_action = [1, chosen_random_slot, "(P) Add a relay next to a random connected relay's neighborhood."]
        free_slots.remove(chosen_random_slot)
        remember_used_relays.append(chosen_random_slot)
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def delete_random_relay(sentinels, sinked_relays, free_slots, remember_used_relays):
    performed_action = []
    min_meshes, max_meshes = get_min_max_meshes(sinked_relays, free_slots)
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
        relays_next_to_chosen_sentinel = [relay for relay in sinked_relays if math.dist(chosen_sentinel, relay[0]) < 30]
        
        # Delete a random relay next to the chosen sentinel
        if relays_next_to_chosen_sentinel:
            chosen_random_relay = random.choice(relays_next_to_chosen_sentinel)
            sinked_relays = [relay for relay in sinked_relays if relay[0] != chosen_random_relay[0]]
            performed_action = [3, chosen_random_relay[0], "(P) Delete a random relay that's next to a sentinel with multiple relay neighbors"]
            free_slots.append(chosen_random_relay[0])
            remember_used_relays.append(chosen_random_relay[0])
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def swap_relays_with_free_slots(sentinels, sinked_relays, free_slots, remember_used_relays):
    performed_action = []
    
    if sinked_relays and free_slots:
        # Choose a relay and a free slot randomly
        relay_index = random.randint(0, len(sinked_relays) - 1)
        free_slot_index = random.randint(0, len(free_slots) - 1)
        
        # Swap the positions of the relay and the free slot
        relay_position = sinked_relays[relay_index][0]
        free_slot_position = free_slots[free_slot_index]
        sinked_relays[relay_index] = (free_slot_position, sinked_relays[relay_index][1])
        free_slots[free_slot_index] = relay_position

        # Update the performed action to reflect the swap
        performed_action = [2, (sinked_relays[relay_index], free_slots[free_slot_index]), "(LS) Swap relay with free slot"]
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def add_relay_next_to_sentinel(sentinels, sinked_relays, free_slots, remember_used_relays, custom_range):
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
                sinked_relays.append([chosen_slot, 1])
                performed_action = [1, chosen_slot, "(P) Add a relay next to a sentinel with no relay neighbors."]
                free_slots.remove(chosen_slot)
                remember_used_relays.append(chosen_slot)
                break  # Only add one relay per iteration
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def swap_relays(sentinels, sinked_relays, free_slots, remember_used_relays):
    performed_action = []
    
    if len(sinked_relays) >= 2:
        # Choose two distinct relays randomly
        i, j = random.sample(range(len(sinked_relays)), 2)
        
        # Swap the positions of the two relays
        sinked_relays[i], sinked_relays[j] = sinked_relays[j], sinked_relays[i]

        # Update the performed action to reflect the swap
        performed_action = [2, (sinked_relays[i], sinked_relays[j]), "(LS) Swap relay positions"]
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

'''
    Helper functions
'''

def get_min_max_meshes(sinked_relays, free_slots):
    min_meshes_candidate = []
    max_meshes_candidate = []
    random.shuffle(sinked_relays)

    for i in range(len(sinked_relays)):
        empty_meshes_counter = 0

        # Calculate meshes around a sinked relay
        for j in range(len(free_slots)):
            if math.dist(sinked_relays[i][0], free_slots[j]) < 30:
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

def Variable_Neighborhood_Descent(grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, lmax, alpha, beta):
    l = 1  # Neighborhood counter
    
    while l <= lmax:
        i = 0  # Neighbor counter
        improvement = True  # Flag to indicate improvement

        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
        previous = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size, alpha, beta)
        
        print(f'Sentinel bman: {sentinel_bman}')
        
        while improvement and i < len(sinked_relays) + 1:
            improvement = False
            i += 1
            
            if l == 1:
                for _ in range(len(free_slots)):
                    free_slots, sinked_relays, action, remember_used_relays = add_next_to_relay(sinked_sentinels, sinked_relays, free_slots, [])
                    print('Relay added')
            
            elif l == 2:
                free_slots, sinked_relays, action, remember_used_relays = delete_random_relay(sinked_sentinels, sinked_relays, free_slots, [])
                print('Random relay deleted ')
            
            elif l == 3:
                free_slots, sinked_relays, action, remember_used_relays = delete_relay_next_to_sentinel(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
                print('Relay next to sentinel deleted')
            
            elif l == 4:
                for _ in range(2):
                    free_slots, sinked_relays, action, remember_used_relays = swap_relays_with_free_slots(sinked_sentinels, sinked_relays, free_slots, [])
                    print('Relays positions swaped')
            
            elif l == 5:
                for _ in range(len(free_slots)):
                    free_slots, sinked_relays, action, remember_used_relays = add_relay_next_to_sentinel(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
                    print('Relay added next to sentinel with no neighbors')
            
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
            after = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size, alpha, beta)

            print(f'Sentinel bman: {sentinel_bman}')

            if previous > after:
                print(f'\nPrevious Fitness: {previous}, After Fitness: {after}')
                improvement = True
        
        l += 1
        print(f'\n {l} Neighborhood Previous Fitness: {previous}, After Fitness: {after}')
        print(f'\nThere are {len(sinked_relays)} relays deployed')
        print(f'There are {len(free_slots)} free slots remaining')

        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
        print(f'VND Sentinel bman: {sentinel_bman}') 

    return sinked_relays, free_slots