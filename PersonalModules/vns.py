import math
import random
from PersonalModules.greedy import greedy_algorithm

from PersonalModules.utilities import bellman_ford, get_stat

def perturbation(sentinels, sinked_relays, free_slots, remember_used_relays, custom_range):
    performed_action = []
    action = random.randint(1, 5)

    # 1: Add next to random connected relay | 2: delete random connected relay | 3: delete next to sentinel ADDRmembring
    if action == 1:
        print('action 1')
        candidate_slots = []
        while len(candidate_slots) == 0:
            allowed_sinked_relays = [x for x in sinked_relays if x not in remember_used_relays]
            min_meshes, max_meshes = get_min_max_meshes(allowed_sinked_relays, free_slots,custom_range)
            chosen_random_relay = max_meshes[0][0]
            for i in range(len(free_slots)):
                if math.dist(chosen_random_relay, free_slots[i]) < custom_range: # the range*
                    candidate_slots.append(free_slots[i])
            remember_used_relays.append(chosen_random_relay)
        chosen_random_relay = candidate_slots[random.randint(0, len(candidate_slots) - 1)]
        sinked_relays.append([chosen_random_relay, 1])
        performed_action = [1, chosen_random_relay, "(P) Add a relay next to a random connected relay's neighborhood."]
        for i in range(len(free_slots)):
            if free_slots[i] == chosen_random_relay:
                iteration = i
        free_slots.pop(iteration)

        '''if action == 1:
        print('action 1')
        candidate_slots = []
        while len(candidate_slots) == 0:
            allowed_sinked_relays = [x for x in sinked_relays if x not in remember_used_relays]
            min_meshes, max_meshes = get_min_max_meshes(allowed_sinked_relays, free_slots, custom_range)
            if max_meshes:
                chosen_random_relay = max_meshes[0][0]
                for i in range(len(free_slots)):
                    if math.dist(chosen_random_relay, free_slots[i]) < custom_range: # the range*
                        candidate_slots.append(free_slots[i])
                remember_used_relays.append(chosen_random_relay)
            else:
                # Handle the case when max_meshes is empty
                print('Max meshes is empty')
                break
                
        if candidate_slots:
            chosen_random_relay = candidate_slots[random.randint(0, len(candidate_slots) - 1)]
            sinked_relays.append([chosen_random_relay, 1])
            performed_action = [1, chosen_random_relay, "(P) Add a relay next to a random connected relay's neighborhood."]
            free_slots.remove(chosen_random_relay)
        else:
            print('Candidate slots is empty')
            print('Performed action:', performed_action)'''

        '''elif action == 2:
        print('action 2')
        min_meshes, max_meshes = get_min_max_meshes(sinked_relays, free_slots, custom_range)
        chosen_random_relay = min_meshes[0][0]
        for i in range(len(sinked_relays)):
            if chosen_random_relay == sinked_relays[i][0]:
                iteration = i
                break
        sinked_relays.pop(iteration)
        performed_action = [2, chosen_random_relay, "(P) Delete a random connected relay."]
        free_slots.append(chosen_random_relay)
        remember_used_relays.append(chosen_random_relay)'''
    
    elif action == 2:
        print('action 2')
        min_meshes, max_meshes = get_min_max_meshes(sinked_relays, free_slots, custom_range)
        if min_meshes:
            chosen_random_relay = min_meshes[0][0]
            for i in range(len(sinked_relays)):
                if chosen_random_relay == sinked_relays[i][0]:
                    sinked_relays.pop(i)
                    break
            performed_action = [2, chosen_random_relay, "(P) Delete a random connected relay."]
            free_slots.append(chosen_random_relay)
            remember_used_relays.append(chosen_random_relay)
            print('Performed action:', performed_action)
        else:
            # Handle the case when min_meshes is empty
            print("Min meshes is empty!")
            print('Performed action:', performed_action)

        '''elif action == 3:
        candidate_slots = []

        for i in range(len(sentinels)):
            for j in range(len(sinked_relays)):
                print("action 3")
                if math.dist(sentinels[i], sinked_relays[j][0]) < custom_range:
                    if sinked_relays[j][0] in candidate_slots:
                        pass
                    else:
                        candidate_slots.append(sinked_relays[j])
        min_meshes, max_meshes = get_min_max_meshes(candidate_slots, free_slots, custom_range)
        chosen_random_relay = min_meshes[0][0]
        for i in range(len(sinked_relays)):
            if sinked_relays[i][0] == chosen_random_relay:
                sinked_relays.pop(i)
                break
        performed_action = [3, chosen_random_relay, "(P) Delete a random relay that's next to a sentinel"]
        free_slots.append(chosen_random_relay)
        remember_used_relays.append(chosen_random_relay)'''
    
    elif action == 3:
        candidate_slots = []
        print("action 3")
        if isinstance(sentinels, (list, tuple)):  # Check if sentinels is iterable
            for i in range(len(sentinels)):
                if isinstance(sentinels[i], (list, tuple)):  # Check if sentinels[i] is iterable
                    for j in range(len(sinked_relays)):
                        if isinstance(sinked_relays[j], (list, tuple)):  # Check if sinked_relays[j] is iterable
                            if math.dist(sentinels[i], sinked_relays[j][0]) < custom_range:
                                if sinked_relays[j][0] in candidate_slots:
                                    pass
                                else:
                                    candidate_slots.append(sinked_relays[j])
                        else:
                            print("Sinked relays[j] is not iterable!")

        min_meshes, max_meshes = get_min_max_meshes(candidate_slots, free_slots, custom_range)
        if min_meshes:
            chosen_random_relay = min_meshes[0][0]
            for i in range(len(sinked_relays)):
                if sinked_relays[i][0] == chosen_random_relay:
                    sinked_relays.pop(i)
                    break
            performed_action = [3, chosen_random_relay, "(P) Delete a random relay that's next to a sentinel"]
            free_slots.append(chosen_random_relay)
            remember_used_relays.append(chosen_random_relay)
            print('Performed action:', performed_action)
        else:
            # Handle the case when min_meshes is empty
            print("Min meshes is empty!")
            print('Performed action:', performed_action)

    elif action == 4:
        print('action 4: Swap the positions of two relays')
        if len(sinked_relays) >= 2:
            # Choose two distinct relays randomly
            i, j = random.sample(range(len(sinked_relays)), 2)
            # Convert the tuples to lists for mutable assignment
            relay_i = list(sinked_relays[i])
            relay_j = list(sinked_relays[j])
            # Swap the positions
            relay_i[0], relay_j[0] = relay_j[0], relay_i[0]
            # Convert back to tuples before assigning back to the list
            sinked_relays[i] = tuple(relay_i)
            sinked_relays[j] = tuple(relay_j)
            performed_action = [4, (sinked_relays[i][0], sinked_relays[j][0]), "(P) Swap the positions of two relays."]

    elif action == 5:
        print('action 5: Move a relay to a different nearby location')
        if sinked_relays:
            chosen_relay = random.choice(sinked_relays)
            candidate_slots = [slot for slot in free_slots if math.dist(slot, chosen_relay[0]) < custom_range]
            if candidate_slots:
                new_position = random.choice(candidate_slots)
                # Convert chosen_relay to a list for mutable assignment
                chosen_relay = list(chosen_relay)
                chosen_relay[0] = new_position
                # Convert chosen_relay back to a tuple before adding it to sinked_relays
                chosen_relay = tuple(chosen_relay)
                performed_action = [5, (chosen_relay[0], new_position), "(P) Move a relay to a different nearby location."]


    return free_slots, sinked_relays, performed_action, remember_used_relays

def perturbation2(sinked_sentinels, sinked_relays, free_slots, remember_used_relays, sink, custom_range, sentinel_bman):
    performed_action = []
    action = random.randint(1, 3)  # Reduce action choices to focus on strategic additions and removals

    if action == 1:
        print('Add relay next to area with poor coverage')
        candidate_slots = []

        # Identify areas with poor coverage
        poor_coverage_slots = [slot for slot in sinked_sentinels if not check_routes_to_sink(sinked_sentinels, sinked_relays, sink, custom_range)]

        if poor_coverage_slots:
            # Choose a random slot with poor coverage
            chosen_slot = random.choice(poor_coverage_slots)

            # Find a nearby relay-free slot to add a relay
            for slot in free_slots:
                if math.dist(slot, chosen_slot) < custom_range:
                    candidate_slots.append(slot)

            if candidate_slots:
                # Choose a nearby slot to add the relay
                chosen_random_relay = random.choice(candidate_slots)
                sinked_relays.append([chosen_random_relay, 1])
                performed_action = [1, chosen_random_relay, "(P) Add a relay next to area with poor coverage"]
                free_slots.remove(chosen_random_relay)
            else:
                print('No nearby relay-free slot found')
        else:
            print('No area with poor coverage found')

    elif action == 2:
        print('Remove redundant relay')
        for relay in sinked_relays:
            # Count the number of other relays close to the current relay
            close_relays_count = sum(1 for other_relay in sinked_relays if other_relay != relay and math.dist(relay[0], other_relay[0]) < custom_range)
            if close_relays_count > 2:
                # Remove the current relay if it has more than two other relays close to it
                sinked_relays.remove(relay)
                free_slots.append(relay[0])  # Add the removed relay's slot back to free slots
                performed_action = [2, relay[0], "(P) Remove a redundant relay"]
                print(f"Removed a redundant relay at {relay[0]}")
                break  # Exit the loop after removing one redundant relay
            else:
                print('NO relay deleted')

    elif action == 3:
        print('Add relay far from existing relays')
        candidate_slots = []

        # Identify relays that are far from existing relays
        far_from_existing_relays = [slot for slot in free_slots if all(math.dist(slot, relay[0]) >= custom_range for relay in sinked_relays)]

        if far_from_existing_relays:
            # Choose a random slot far from existing relays
            chosen_slot = random.choice(far_from_existing_relays)

            # Find a nearby relay-free slot to add a relay
            for slot in free_slots:
                if math.dist(slot, chosen_slot) < custom_range:
                    candidate_slots.append(slot)

            if candidate_slots:
                # Choose a nearby slot to add the relay
                chosen_random_relay = random.choice(candidate_slots)
                sinked_relays.append([chosen_random_relay, 1])
                performed_action = [3, chosen_random_relay, "(P) Add a relay far from existing relays"]
                free_slots.remove(chosen_random_relay)
            else:
                print('No nearby relay-free slot found')
        else:
            print('No slot far from existing relays found')

    return free_slots, sinked_relays, performed_action, remember_used_relays

def ls(sentinels, sinked_relays, free_slots, action, remember_used_relays, custom_range):
    performed_action = []
    action = random.randint(1, 5)
    # 1: Add next to random connected relay | 2: delete random connected relay | 3: delete next to sentinel ADDRmembring
    if action == 1:
        '''print('LS Action 1')
        candidate_slots = []
        while len(candidate_slots) == 0:
            allowed_sinked_relays = [x for x in sinked_relays if x not in remember_used_relays]
            min_meshes, max_meshes = get_min_max_meshes(allowed_sinked_relays, free_slots, custom_range)
            chosen_random_relay = max_meshes[0][0]
            for i in range(len(free_slots)):
                if math.dist(chosen_random_relay, free_slots[i]) < custom_range:
                    candidate_slots.append(free_slots[i])
            remember_used_relays.append(chosen_random_relay)
        chosen_random_relay = candidate_slots[random.randint(0, len(candidate_slots) - 1)]
        sinked_relays.append([chosen_random_relay, 1])
        performed_action = [1, chosen_random_relay, "(LS) Add a relay next to a random connected relay's neighborhood."]
        for i in range(len(free_slots)):
            if free_slots[i] == chosen_random_relay:
                iteration = i
        free_slots.pop(iteration)'''
        
        print('LS Action 1')
        candidate_slots = []
        while len(candidate_slots) == 0:
            allowed_sinked_relays = [x for x in sinked_relays if x not in remember_used_relays]
            min_meshes, max_meshes = get_min_max_meshes(allowed_sinked_relays, free_slots, custom_range)
            
            if max_meshes:  # Check if max_meshes is not empty
                chosen_random_relay = max_meshes[0][0]
                for i in range(len(free_slots)):
                    if math.dist(chosen_random_relay, free_slots[i]) < custom_range:
                        candidate_slots.append(free_slots[i])
                remember_used_relays.append(chosen_random_relay)
            else:
                # Handle the case when max_meshes is empty
                print('Max meshes is empty')
                break
            
        if candidate_slots:
            chosen_random_relay = candidate_slots[random.randint(0, len(candidate_slots) - 1)]
            sinked_relays.append([chosen_random_relay, 1])
            performed_action = [1, chosen_random_relay, "(LS) Add a relay next to a random connected relay's neighborhood."]
            free_slots.remove(chosen_random_relay)
        else:
            print('Candidate slots is empty')
    
        '''elif action == 2:
        print('LS Action 2')
        min_meshes, max_meshes = get_min_max_meshes(sinked_relays, free_slots, custom_range)
        chosen_random_relay = min_meshes[0][0]
        for i in range(len(sinked_relays)):
            if chosen_random_relay == sinked_relays[i][0]:
                iteration = i
                break
        sinked_relays.pop(iteration)
        performed_action = [2, chosen_random_relay, "(LS) Delete a random connected relay."]
        free_slots.append(chosen_random_relay)
        remember_used_relays.append(chosen_random_relay)'''
    elif action == 2:
        print('LS Action 2')
        min_meshes, max_meshes = get_min_max_meshes(sinked_relays, free_slots, custom_range)
        
        if min_meshes:  # Check if min_meshes is not empty
            chosen_random_relay = min_meshes[0][0]
            for i in range(len(sinked_relays)):
                if chosen_random_relay == sinked_relays[i][0]:
                    iteration = i
                    break
            sinked_relays.pop(iteration)
            performed_action = [2, chosen_random_relay, "(LS) Delete a random connected relay."]
            free_slots.append(chosen_random_relay)
            remember_used_relays.append(chosen_random_relay)
        else:
            # Handle the case when min_meshes is empty
            print("Min meshes is empty!")
    
    elif action == 3:
        print('LS Action 3: Delete a random relay that\'s next to a sentinel')
        candidate_slots = []
        for i in range(len(sentinels)):
            for j in range(len(sinked_relays)):
                if math.dist(sentinels[i], sinked_relays[j][0]) < custom_range:
                    candidate_slots.append(sinked_relays[j])
        if candidate_slots:  # Check if candidate_slots is not empty
            min_meshes, max_meshes = get_min_max_meshes(candidate_slots, free_slots, custom_range)
            if min_meshes:
                chosen_random_relay = min_meshes[0][0]
                for i in range(len(sinked_relays)):
                    if sinked_relays[i][0] == chosen_random_relay:
                        sinked_relays.pop(i)
                        break
                performed_action = [3, chosen_random_relay, "(LS) Delete a random relay that's next to a sentinel"]
                free_slots.append(chosen_random_relay)
                remember_used_relays.append(chosen_random_relay)
            else:
                print("Min meshes is empty!")
        else:
            print("Candidate slots is empty!")

    elif action == 4:
        print('LS Action 4: Swap the positions of two relays')
        if len(sinked_relays) >= 2:
            i, j = random.sample(range(len(sinked_relays)), 2)
            # Convert the tuples to lists for mutable assignment
            relay_i = list(sinked_relays[i])
            relay_j = list(sinked_relays[j])
            # Swap positions
            relay_i[0], relay_j[0] = relay_j[0], relay_i[0]
            # Convert back to tuples before assigning them back
            sinked_relays[i] = tuple(relay_i)
            sinked_relays[j] = tuple(relay_j)
            performed_action = [4, (relay_i[0], relay_j[0]), "(LS) Swap the positions of two relays."]

    elif action == 5:
        print('LS Action 5: Move a relay to a different nearby location')
        if sinked_relays:
            chosen_relay = random.choice(sinked_relays)
            candidate_slots = [slot for slot in free_slots if math.dist(slot, chosen_relay[0]) < custom_range]
            if candidate_slots:
                new_position = random.choice(candidate_slots)
                # Convert the tuple to a list for mutable assignment
                chosen_relay = list(chosen_relay)
                chosen_relay[0] = new_position
                # Convert back to a tuple before assigning it back
                chosen_relay = tuple(chosen_relay)
                performed_action = [5, (chosen_relay[0], new_position), "(LS) Move a relay to a different nearby location."]


    return free_slots, sinked_relays, performed_action

def ls2(sentinels, sinked_relays, free_slots, action, remember_used_relays, custom_range):
    performed_action = []

    if action == 1:
        print('Add relay to improve connectivity')
        candidate_slots = []

        # Identify relays that are far from existing relays
        far_from_existing_relays = [slot for slot in free_slots if all(math.dist(slot, relay[0]) >= custom_range for relay in sinked_relays)]

        if far_from_existing_relays:
            # Choose a random slot far from existing relays
            chosen_slot = random.choice(far_from_existing_relays)

            # Find a nearby relay-free slot to add a relay
            for slot in free_slots:
                if math.dist(slot, chosen_slot) < custom_range:
                    candidate_slots.append(slot)

            if candidate_slots:
                # Choose a nearby slot to add the relay
                chosen_random_relay = random.choice(candidate_slots)
                sinked_relays.append([chosen_random_relay, 1])
                performed_action = [1, chosen_random_relay, "(LS) Add relay to improve connectivity"]
                free_slots.remove(chosen_random_relay)
            else:
                print('No nearby relay-free slot found')
        else:
            print('No slot far from existing relays found')

    elif action == 2:
        print('Swap relay positions to optimize coverage and reduce redundancy')
        if len(sinked_relays) >= 2:
            # Choose two distinct relays randomly
            i, j = random.sample(range(len(sinked_relays)), 2)

            # Swap the positions of the two relays
            sinked_relays[i][0], sinked_relays[j][0] = sinked_relays[j][0], sinked_relays[i][0]

            performed_action = [2, (sinked_relays[i][0], sinked_relays[j][0]), "(LS) Swap relay positions"]

    elif action == 3:
        print('Move relay to optimize network connectivity or fill coverage gaps')
        if sinked_relays:
            # Choose a random relay to move
            chosen_relay = random.choice(sinked_relays)

            # Identify nearby slots to move the relay
            candidate_slots = [slot for slot in free_slots if math.dist(slot, chosen_relay[0]) < custom_range]

            if candidate_slots:
                # Choose a nearby slot to move the relay
                new_position = random.choice(candidate_slots)
                # Convert chosen_relay to a list for mutable assignment
                chosen_relay = list(chosen_relay)
                chosen_relay[0] = new_position  # Update the position of the relay
                # Convert chosen_relay back to a tuple before adding it to sinked_relays
                chosen_relay = tuple(chosen_relay)
                performed_action = [3, (chosen_relay[0], new_position), "(LS) Move relay to optimize connectivity or fill coverage gaps"]

    return free_slots, sinked_relays, performed_action

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
                    print('min mesh added')

                # Acquire maximum meshes
                if max_meshes_candidate[1] < empty_meshes_counter:
                    max_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
                    print('max mesh added')
            else:
                min_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
                max_meshes_candidate = [sinked_relays[i], empty_meshes_counter]

    return min_meshes_candidate, max_meshes_candidate

def check_routes_to_sink(sinked_sentinels, sinked_relays, sink, custom_range):
    def has_route_to_sink(start, visited):
        if start == sink:
            return True
        visited.add(start)
        for relay, hops in sinked_relays:
            if math.dist(start, relay) <= custom_range and relay not in visited:
                if has_route_to_sink(relay, visited):
                    return True
        return False

    for sentinel in sinked_sentinels:
        visited = set()
        if not has_route_to_sink(sentinel, visited):
            return False
    return True

'''def has_node_nearby(sinked_sentinels, sinked_relays, custom_range):
    for sentinel in sinked_sentinels:
        sentinel_position = sentinel[0]  # Assuming the position is stored as the first element of the tuple
        for relay, _ in sinked_relays:
            if isinstance(relay, (list, tuple)) and isinstance(sentinel_position, (list, tuple)):
                if math.dist(sentinel_position, relay) < custom_range:
                    return True
    return False'''

def has_node_nearby(sentinel_bman):
    if 999 in sentinel_bman:
        return True
    else:
        return False

def Variable_Neighborhood_Search(grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range):
    pre_sinked_relays = sinked_relays[:]
    pre_free_slots = free_slots[:]
    remember_used_relays = []
    First_time = True
    Error = False
    Err_counter = 0
    P_Err_counter = 0
    LS_Err_counter = 0
    INF = float('inf')
    consecutive_errors = 0
    stop = False
    repitition = 0

    distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
    
    while consecutive_errors != 4 or First_time == True or Error == True or repitition <= 10:
        # Bellman Ford
        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
        
        # Actual function
        previous, relays_before, hops_before = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, 0.5, 0.5)
        a = len(sinked_relays)

        allowed_sinked_relays = [x for x in sinked_relays if x not in remember_used_relays]
        if len(allowed_sinked_relays) != 0:
            if First_time:
                First_time = False
                print("\n   VNS Perturbation has started.")
            
            free_slots, sinked_relays, performed_action, remember_used_relays = perturbation2(sinked_sentinels, sinked_relays, 
                                                                                              free_slots, remember_used_relays, sink, custom_range, sentinel_bman)

            # print("\nPerformed action\n", performed_action[2])
            print("Before: ", a)
            print("After: ", len(sinked_relays))

            # Bellman Ford
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays,
                                                                  sinked_sentinels)

            # back again
            current, relays_before, hops_before = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, 0.5, 0.5)
            print("Now: ", current, " VS Then: ", previous, "\nSentinel Bman: ", sentinel_bman)
            for i in range(len(sentinel_bman)):
                if sentinel_bman[i] == INF or previous < current:
                    Error = True
                    break
                else:
                    Error = False

            print('P iteration number:', repitition, '\n')
            repitition +=1

            if Error:
                sinked_relays = pre_sinked_relays[:]
                free_slots = pre_free_slots[:]
                Err_counter = Err_counter + 1
                P_Err_counter = P_Err_counter + 1
                print("\nP Error number: ", P_Err_counter, "\n")
                consecutive_errors = consecutive_errors + 1
            else:
                consecutive_errors = 4
            if consecutive_errors == 4:
                Error = False

    print("\n   VNS Perturbation finished execution !")
    action = random.randint(1, 5)
    # ------------------------------------------------------------------------------------------------------------------
    First_time = True
    Error = False
    ls_pre_sinked_relays = sinked_relays[:]
    ls_pre_free_slots = free_slots[:]
    iteration = 0
    consecutive_errors = 0
    while iteration != 20 or First_time == True:
        if action == 4:
            action = 1
        iteration = iteration + 1

        # Bellman Ford
        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)

        # Actual function
        previous, relays_before, hops_before = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, 0.5, 0.5)
        a = len(sinked_relays)

        if First_time:
            First_time = False
            print("\n   Local Search algorithm has started.")
        free_slots, sinked_relays, performed_action = ls(sinked_sentinels, sinked_relays, free_slots, action,
                                                         remember_used_relays, custom_range)

        # print("\nPerformed action", performed_action[2])
        print("Before: ", a)
        print("After: ", len(sinked_relays))

        # Bellman Ford
        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)

        # back again
        current, relays_before, hops_before = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, 0.5, 0.5)
        print("Now: ", current, " VS Then: ", previous, "\nSentinel Bman: ", sentinel_bman)
        for i in range(len(sentinel_bman)):
            if sentinel_bman[i] == INF or previous < current:
                Error = True
                break
            else:
                Error = False
        if Error:
            sinked_relays = ls_pre_sinked_relays[:]
            free_slots = ls_pre_free_slots[:]
            Err_counter = Err_counter + 1
            LS_Err_counter = LS_Err_counter + 1
            print("\nLS Error number: ", LS_Err_counter, "\n")
            action = action + 1
            iteration = iteration # - 1
            consecutive_errors = consecutive_errors + 1
        else:
            consecutive_errors = 0
            Error = False
        if previous >= current:
            ls_pre_sinked_relays = sinked_relays[:]
            ls_pre_free_slots = free_slots[:]
        '''if consecutive_errors == 4:
            iteration = 10
            Error = False'''

    print("\n   Local Search algorithm finished executing !")
    print("\n Total Errors: ", Err_counter)
    return sinked_relays, free_slots
