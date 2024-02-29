import math
import random
import multiprocessing

from PersonalModules.utilities import bellman_ford, epsilon_constraints, get_stat

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
    
    sentinel_neighbor_count = {tuple(sentinel): 0 for sentinel in sentinels}
    
    for sentinel in sentinels:
        for relay in sinked_relays:
            if math.dist(sentinel, relay[0]) < custom_range:
                sentinel_neighbor_count[tuple(sentinel)] += 1
    
    sentinels_with_multiple_neighbors = [sentinel for sentinel, count in sentinel_neighbor_count.items() if count > 1]
    
    if sentinels_with_multiple_neighbors:
        chosen_sentinel = random.choice(sentinels_with_multiple_neighbors)
        relays_next_to_chosen_sentinel = [relay for relay in sinked_relays if math.dist(chosen_sentinel, relay[0]) < 30]
        
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
        relay_index = random.randint(0, len(sinked_relays) - 1)
        free_slot_index = random.randint(0, len(free_slots) - 1)
        
        relay_position = sinked_relays[relay_index][0]
        free_slot_position = free_slots[free_slot_index]
        sinked_relays[relay_index] = (free_slot_position, sinked_relays[relay_index][1])
        free_slots[free_slot_index] = relay_position

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
                break
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def get_min_max_meshes(sinked_relays, free_slots):
    min_meshes_candidate = []
    max_meshes_candidate = []
    random.shuffle(sinked_relays)

    for i in range(len(sinked_relays)):
        empty_meshes_counter = 0

        for j in range(len(free_slots)):
            if math.dist(sinked_relays[i][0], free_slots[j]) < 30:
                empty_meshes_counter = empty_meshes_counter + 1

        if len(min_meshes_candidate) != 0 and len(max_meshes_candidate) != 0:
            if min_meshes_candidate[1] > empty_meshes_counter:
                min_meshes_candidate = [sinked_relays[i], empty_meshes_counter]

            if max_meshes_candidate[1] < empty_meshes_counter:
                max_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
        else:
            min_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
            max_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
    
    return min_meshes_candidate, max_meshes_candidate

def parallel_explore_neighborhood(task_queue, result_queue, grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, alpha, beta):
    while not task_queue.empty():
        task = task_queue.get()
        neighborhood_index = task[0]
        
        if neighborhood_index == 1:
            free_slots, sinked_relays, action, remember_used_relays = add_next_to_relay(sinked_sentinels, sinked_relays, free_slots, [])
        elif neighborhood_index == 2:
            free_slots, sinked_relays, action, remember_used_relays = delete_random_relay(sinked_sentinels, sinked_relays, free_slots, [])
        elif neighborhood_index == 3:
            free_slots, sinked_relays, action, remember_used_relays = delete_relay_next_to_sentinel(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
        elif neighborhood_index == 4:
            free_slots, sinked_relays, action, remember_used_relays = swap_relays_with_free_slots(sinked_sentinels, sinked_relays, free_slots, [])
        elif neighborhood_index == 5:
            free_slots, sinked_relays, action, remember_used_relays = add_relay_next_to_sentinel(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
        
        result_queue.put((neighborhood_index, free_slots, sinked_relays, action, remember_used_relays))

def parallel_vnd(grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, lmax, alpha, beta):
    num_cores = multiprocessing.cpu_count()
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    for neighborhood_index in range(1, 6):
        task_queue.put((neighborhood_index,))

    processes = []
    for _ in range(num_cores):
        process = multiprocessing.Process(target=parallel_explore_neighborhood, args=(task_queue, result_queue, grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, alpha, beta))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    while not result_queue.empty():
        neighborhood_index, result = result_queue.get()
        
        if neighborhood_index == 1:
            free_slots, sinked_relays, action, remember_used_relays = result
        elif neighborhood_index == 2:
            free_slots, sinked_relays, action, remember_used_relays = result
        elif neighborhood_index == 3:
            free_slots, sinked_relays, action, remember_used_relays = result
        elif neighborhood_index == 4:
            free_slots, sinked_relays, action, remember_used_relays = result
        elif neighborhood_index == 5:
            free_slots, sinked_relays, action, remember_used_relays = result

    return sinked_relays, free_slots
