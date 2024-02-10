import math
import random

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from PersonalModules.utilities import display_realtime


def closest_relay(sink, candidate, sinked_relays, forbidden_sinked_relays):
    # Here we're trying to find the closest sinked_relay to candidate and if tied then choose the sinked_relay
    # closest to sink
    counter = 0
    closest = []
    if len(forbidden_sinked_relays) == 0:
        min = math.dist(candidate, sinked_relays[0][0])
        closest = sinked_relays[0]
        for i in range(len(sinked_relays)):

            # Find the closest relay
            if math.dist(candidate, sinked_relays[i][0]) < min:
                min = math.dist(candidate, sinked_relays[i][0])
                closest = sinked_relays[i]

            # If tied choose closest to sink
            elif math.dist(candidate, sinked_relays[i][0]) == min:
                if math.dist(sink, sinked_relays[i][0]) < math.dist(sink, closest[0]):
                    min = math.dist(sink, sinked_relays[i][0])
                    closest = sinked_relays[i]
    elif len(forbidden_sinked_relays) != 0:
        allowed_sinked_relays = [x for x in sinked_relays if x not in forbidden_sinked_relays]
        min = math.dist(candidate, allowed_sinked_relays[0][0])
        closest = allowed_sinked_relays[0]
        for i in range(len(allowed_sinked_relays)):

            # Find the closest relay
            if math.dist(candidate, allowed_sinked_relays[i][0]) < min:
                min = math.dist(candidate, allowed_sinked_relays[i][0])
                closest = allowed_sinked_relays[i]

            # If tied choose closest to sink
            elif math.dist(candidate, allowed_sinked_relays[i][0]) == min:
                if math.dist(sink, allowed_sinked_relays[i][0]) < math.dist(sink, closest[0]):
                    min = math.dist(sink, allowed_sinked_relays[i][0])
                    closest = allowed_sinked_relays[i]
        allowed_sinked_relays = [x for x in sinked_relays if x not in forbidden_sinked_relays]
        closest_candidates = []

        # Find the closest relay
        for i in range(len(allowed_sinked_relays)):
            if math.dist(candidate, allowed_sinked_relays[i][0]) < 30: # This is the range*
                closest_candidates.append(allowed_sinked_relays[i])

        if len(closest_candidates) > 1:
            closest = closest_candidates[0]
            min = closest_candidates[0][1]

            # Choose the relay with minimum number of hops to sink
            for i in range(len(closest_candidates)):
                if min > closest_candidates[i][1]:
                    closest = closest_candidates[i]
                    min = closest_candidates[i][1]

    return closest, counter


def greedy_algorithm(sink, sinkless_sentinels, free_slots, max_hops_number, custom_range):
    First_time = True
    sentinels = sinkless_sentinels[:]
    grid = len(free_slots) + len(sinkless_sentinels) + 1
    print("The grid =", grid)
    sinked_sentinels, sinked_relays, sinkless_relays, candidate_slots = [], [], [], []
    found_forbidden, ERROR = False, False

    # THIS IS A "while loop" FOR CHOOSING NEXT SENTINEL!!!
    while len(sinkless_sentinels) != 0:
        remember_free_slots = free_slots
        forbidden_sinked_relays = []
        forbidden, stop = False, False
        number_of_hops, extra_hops = 0, 0
        current_sinkless_sentinel = sinkless_sentinels[random.randint(0, len(sinkless_sentinels) - 1)]
        current_node = current_sinkless_sentinel

        # If current sentinel already has sink or sinked_relay nearby
        if math.dist(current_sinkless_sentinel, sink) < custom_range:
            stop = True
        if len(sinked_relays) != 0:
            a, useless_variable = closest_relay(sink, current_sinkless_sentinel, sinked_relays,
                                                forbidden_sinked_relays)
            if math.dist(current_sinkless_sentinel, a[0]) < custom_range:  # and max_hops_number >= a[1]
                stop = True

        # THIS IS A "while loop" FOR CHOOSING NEXT NEIGHBOR!!!
        while not stop:

            # Determine the nearby empty slots near the current_node then assign them to candidate_slots
            for j in range(len(free_slots)):
                if math.dist(current_node, free_slots[j]) < custom_range:
                    candidate_slots.append(free_slots[j])

            # If there is MULTIPLE candidates then choose the best among them
            if len(candidate_slots) >= 1:
                chosen_node, forbidden, forbidden_sinked_relays, extra_hops = best_neighbor(sink, sinked_relays,
                                                                                            candidate_slots,
                                                                                            free_slots,
                                                                                            current_sinkless_sentinel,
                                                                                            max_hops_number,
                                                                                            number_of_hops,
                                                                                            forbidden_sinked_relays)
                temp = []
                for i in range(len(free_slots)):
                    if free_slots[i] != chosen_node:
                        temp.append(free_slots[i])
                free_slots = temp
            elif len(candidate_slots) == 0 and math.dist(current_node, current_sinkless_sentinel) < custom_range:
                if len(sinked_relays) != 0 and len(sinked_relays) != len(forbidden_sinked_relays):
                    a, useless_variable = closest_relay(sink, current_node, sinked_relays, forbidden_sinked_relays)
                    extra_hops = a[1]
                stop = True

            # Calculate the number of hops
            number_of_hops += 1

            sinkless_relays.append([chosen_node, number_of_hops])

            # Reset the candidates
            candidate_slots = []

            # The chosen_node will become current_node (who'll search for his best candidates)
            current_node = chosen_node

            # If that route exceeds the number of max hops (forbidden is TRUE) then search again from the start (with
            # resetting everything) and remember the forbidden sinked relay, otherwise stop
            if forbidden:
                current_node = current_sinkless_sentinel
                sinkless_relays = []
                number_of_hops = 0
                free_slots = remember_free_slots
            elif not forbidden:

                # Condition to STOP the loop when near sink or sinked_relay
                if len(sinked_relays) != 0 and len(sinked_relays) != len(forbidden_sinked_relays):
                    a, useless_variable = closest_relay(sink, current_node, sinked_relays, forbidden_sinked_relays)
                if (len(sinked_relays) == 0 or (
                        len(sinked_relays) != 0 and len(sinked_relays) == len(forbidden_sinked_relays))) and math.dist(
                    current_node, sink) < 30:
                    stop = True
                if (len(sinked_relays) != 0 and len(sinked_relays) != len(forbidden_sinked_relays)) and (
                        math.dist(current_node, a[0]) < 30 or math.dist(current_node, sink) < 30):
                    stop = True
            # display_realtime(grid, sink, sinked_relays, sinkless_sentinels)

        if ERROR:
            break
        if not found_forbidden:

            # Remove sinked_sentinels from sinkless_sentinels
            temp = []
            sinked_sentinels.append(current_sinkless_sentinel)
            for i in range(len(sinkless_sentinels)):
                if sinkless_sentinels[i] != current_sinkless_sentinel:
                    temp.append(sinkless_sentinels[i])
            sinkless_sentinels = temp

            # Remove sinked_relays from sinkless_relays
            if len(sinkless_relays) != 0:
                for i in range(len(sinkless_relays)):
                    sinkless_relays[i][1] = (number_of_hops + 1) - sinkless_relays[i][1] + extra_hops
                    sinked_relays.append(sinkless_relays[i])
                sinkless_relays = []

        # Another Display
        if len(sinked_relays) != 0:
            '''display(grid, sink, sinked_relays, sinked_sentinels, sinkless_relays, sinkless_sentinels,
                    current_sinkless_sentinel, current_node, chosen_node,
                    candidate_slots, forbidden_sinked_relays)'''
    Finished = True

    print('\nSinked Sentinels\n',sinked_sentinels)
    print('\nSinked Relays\n', sinked_relays)    
    
    return sinked_sentinels, sinked_relays, free_slots, Finished, ERROR

def best_neighbor(sink, sinked_relays, candidate_slots, free_slots, current_sinkless_sentinel, max_hops_number,
                  number_of_hops, forbidden_sinked_relays):
    counter, extra_hops = 0, 0
    last_sinked_relay, last_sink, last_relay, last_relay_forbidden, forbidden, is_chosen_node_sink, found_forbidden = False, False, False, False, False, False, False
    counter_array, index_array = [], []

    # Calculate how much free_slots each candidate have
    for i in range(len(candidate_slots)):
        for j in range(len(free_slots)):

            # In case this is the first attempt (where sinked_relays == 0) or all sinked_relays are forbidden
            if (len(sinked_relays) == 0 or (len(sinked_relays) != 0 and len(forbidden_sinked_relays) != 0 and len(
                    forbidden_sinked_relays) == len(sinked_relays))) and math.dist(candidate_slots[i],
                                                                                   free_slots[j]) < 30 and math.dist(
                candidate_slots[i], free_slots[j]) != 0:

                # Determine how many free_slots each candidate have (that's far from sink)
                if math.dist(sink, candidate_slots[i]) > 30:
                    counter += 1

                # If a candidate have a nearby covered sink then we declare that there is at least a candidate who
                # arrived
                elif math.dist(sink, candidate_slots[i]) < 30:
                    last_sinked_relay = True
                    last_sink = True
                    counter += 1

            # In case this is NOT the first attempt (where sinked_relays != 0)
            elif len(sinked_relays) != 0 and math.dist(candidate_slots[i], free_slots[j]) < 30 and math.dist(
                    candidate_slots[i], free_slots[j]) != 0:
                a, useless_variable = closest_relay(sink, candidate_slots[i], sinked_relays,
                                                    forbidden_sinked_relays)

                # Determine how many free_slots each candidate have (that's far from sink and allowed sinked_relay)
                if math.dist(sink, candidate_slots[i]) > 30 and math.dist(candidate_slots[i], a[0]) > 30:
                    counter += 1

                # If a candidate have a nearby covered sink then we declare that there is at least a
                # candidate who arrived
                elif math.dist(sink, candidate_slots[i]) < 30 < math.dist(candidate_slots[i], a[0]):
                    last_sinked_relay = True
                    last_sink = True
                    counter += 1

                # If a candidate have a nearby covered sinked_relay then we declare that there is at least a
                # candidate who arrived
                elif math.dist(sink, candidate_slots[i]) > 30 > math.dist(candidate_slots[i], a[0]):
                    if len(forbidden_sinked_relays) == 0:
                        last_sinked_relay = True
                        last_relay = True
                    elif len(forbidden_sinked_relays) != 0:
                        a, function_counter = closest_relay(sink, candidate_slots[i], sinked_relays,
                                                            forbidden_sinked_relays)
                        for k in range(len(forbidden_sinked_relays)):
                            if math.dist(candidate_slots[i], forbidden_sinked_relays[k][0]) < 30:
                                last_relay_forbidden = True
                        if last_relay_forbidden == False or (
                                last_relay_forbidden == True and math.dist(candidate_slots[i], a[0]) < 30):
                            last_sinked_relay = True
                            last_relay = True
                        elif last_relay_forbidden == True and math.dist(candidate_slots[i], a[0]) > 30:
                            last_relay = True

                    counter += 1

                # If a candidate have a nearby covered sink and sinked_relay then we declare that there is at least a
                # candidate who arrived
                elif math.dist(sink, candidate_slots[i]) < 30 and math.dist(candidate_slots[i], a[0]) < 30:
                    a, function_counter = closest_relay(sink, candidate_slots[i], sinked_relays,
                                                        forbidden_sinked_relays)
                    last_sinked_relay = True
                    last_sink = True
                    last_relay = True
                    counter += 1

        counter_array.append(counter)
        counter = 0

    # Is at least one of the candidates near a sink or sinked_relay? "last_sinked_relay == False" means NOPE
    if not last_sinked_relay:

        # Choose the candidate with the highest value of free_slots or if all connected relays are forbidden then
        # choose any slot that's closest to sink
        if len(sinked_relays) != 0 and len(sinked_relays) == len(forbidden_sinked_relays):
            for i in range(len(counter_array)):
                index_array.append(i)
        else:
            max = counter_array[0]
            for i in range(len(counter_array)):
                if max < counter_array[i]:
                    max = counter_array[i]
                    index_array = [i]
                elif max == counter_array[i]:
                    index_array.append(i)

        # If there is more than 2 candidates with the same amount of free_slots then choose closest to sink or
        # closest to sinked_relay
        if len(index_array) > 1:
            if len(sinked_relays) == 0 or (
                    len(sinked_relays) != 0 and len(sinked_relays) == len(forbidden_sinked_relays)):
                min = math.dist(candidate_slots[index_array[0]], sink)
                chosen_node = candidate_slots[index_array[0]]
                for i in range(1, len(index_array)):
                    if math.dist(candidate_slots[index_array[i]], sink) < min:
                        min = math.dist(candidate_slots[index_array[i]], sink)
                        chosen_node = candidate_slots[index_array[i]]
            elif len(sinked_relays) > 0:

                # min1 is minimum distance between candidate and sink
                min1 = math.dist(candidate_slots[index_array[0]], sink)

                # min2 is minimum distance between candidate and sinked_relays
                a, useless_variable = closest_relay(sink, candidate_slots[index_array[0]], sinked_relays,
                                                    forbidden_sinked_relays)
                min2 = math.dist(candidate_slots[index_array[0]], a[0])
                chosen_sinked_relay = a

                chosen_node = candidate_slots[index_array[0]]
                current_chosen_candidate = candidate_slots[index_array[0]]

                # Loop until you find best closest sinked relay
                for i in range(len(index_array)):

                    # Choose the best candidate that's near the closest sinked_relay
                    a, useless_variable = closest_relay(sink, candidate_slots[index_array[i]], sinked_relays,
                                                        forbidden_sinked_relays)
                    if math.dist(candidate_slots[index_array[i]], a[0]) < min2:
                        min2 = math.dist(candidate_slots[index_array[i]], a[0])
                        current_chosen_candidate = candidate_slots[index_array[i]]
                        chosen_sinked_relay = a

                    # If there is more than 2 sinked_relays with the same distance choose the one closest to sink
                    elif math.dist(candidate_slots[index_array[i]], a[0]) == min2:
                        if math.dist(sink, a[0]) < math.dist(sink, chosen_sinked_relay[0]):
                            min2 = math.dist(candidate_slots[index_array[i]], a[0])
                            current_chosen_candidate = candidate_slots[index_array[i]]
                            chosen_sinked_relay = a

                chosen_node = current_chosen_candidate
                # Compare the sinked_relay distance of the current_chosen_candidate with all distances of candidates
                # with sink
                for i in range(len(index_array)):
                    if math.dist(sink, candidate_slots[index_array[i]]) <= min2:
                        chosen_node = candidate_slots[index_array[i]]
                        min2 = math.dist(sink, candidate_slots[index_array[i]])

        elif len(index_array) == 1:
            chosen_node = candidate_slots[index_array[0]]
        number_of_hops += 1
        forbidden_sinked_relays, found_forbidden = intelligent_forbidden(sink,
                                                                         sinked_relays,
                                                                         forbidden_sinked_relays,
                                                                         max_hops_number, number_of_hops)
        if found_forbidden:
            forbidden = True
    # Are the candidates near a sink or sinked_relay? "last_sinked_relay == True" means YES
    elif last_sinked_relay:
        index = 0
        sinked_candidates, sinked_counter_array = [], []

        # If candidate_slots are near a sink ONLY then assign them to sinked_candidates
        if last_sink == True and last_relay == False:
            is_chosen_node_sink = True
            for i in range(len(candidate_slots)):
                if math.dist(sink, candidate_slots[i]) < 30:
                    sinked_candidates.append(candidate_slots[i])
                    sinked_counter_array.append(counter_array[i])

            # Choose the sinked_candidate with the highest value of free_slots (if tied choose closest to sink in
            # terms of distance)
            max = sinked_counter_array[0]
            min_distance = sinked_candidates[0]
            for i in range(len(sinked_counter_array)):
                if max < sinked_counter_array[i]:
                    max = sinked_counter_array[i]
                    min_distance = sinked_candidates[i]
                    index = i
                elif max == sinked_counter_array[i] and math.dist(sink, min_distance) > math.dist(sink,
                                                                                                  sinked_candidates[i]):
                    max = sinked_counter_array[i]
                    min_distance = sinked_candidates[i]
                    index = i

        # If candidate_slots are near a sinked_relays ONLY then assign them to sinked_candidates
        elif last_sink == False and last_relay == True:
            for i in range(len(candidate_slots)):
                a, useless_variable = closest_relay(sink, candidate_slots[i], sinked_relays,
                                                    forbidden_sinked_relays)
                if math.dist(a[0], candidate_slots[i]) < 30:
                    sinked_candidates.append(candidate_slots[i])
                    sinked_counter_array.append(counter_array[i])

            # Choose the sinked_candidate with the highest value of free_slots (if tied then choose candidate closest to
            # sink or sinked relay)
            max = sinked_counter_array[0]
            min_distance = sinked_candidates[0]

            for i in range(len(sinked_counter_array)):
                if max < sinked_counter_array[i]:
                    max = sinked_counter_array[i]
                    index = i
                elif max == sinked_counter_array[i] and len(sinked_counter_array) == 3 and math.dist(
                        sinked_candidates[i], current_sinkless_sentinel) < 45:
                    max = sinked_counter_array[1]
                    index = 1
                elif max == sinked_counter_array[i]:
                    if math.dist(sink, sinked_candidates[i]) < math.dist(sink, min_distance):
                        min_distance = sinked_candidates[i]
                        max = sinked_counter_array[i]
                        index = i
            a, useless_variable = closest_relay(sink, sinked_candidates[index], sinked_relays,
                                                forbidden_sinked_relays)

        # If candidate_slots are near a sink AND sinked_relays then assign them to sinked_candidates
        elif last_sink == True and last_relay == True:
            for i in range(len(candidate_slots)):
                if math.dist(sink, candidate_slots[i]) < 30:
                    sinked_candidates.append(candidate_slots[i])
                    sinked_counter_array.append(counter_array[i])

            # Choose the sinked_candidate with the highest value of free_slots (if tied choose closest to sink in
            # terms of distance)
            max = sinked_counter_array[0]
            min_distance = sinked_candidates[0]
            for i in range(len(sinked_counter_array)):
                if max < sinked_counter_array[i]:
                    max = sinked_counter_array[i]
                    min_distance = sinked_candidates[i]
                    index = i
                elif max == sinked_counter_array[i] and math.dist(sink, min_distance) > math.dist(sink,
                                                                                                  sinked_candidates[i]):
                    max = sinked_counter_array[i]
                    min_distance = sinked_candidates[i]
                    index = i
        chosen_node = sinked_candidates[index]
        if len(sinked_relays) != 0 and ('a' in locals() or 'a' in globals()):
            if last_sink == False and last_relay == True:
                extra_hops = a[1]
            elif last_sink:
                extra_hops = 0
    number_of_hops += 1
    forbidden_sinked_relays, found_forbidden = intelligent_forbidden(sink,
                                                                     sinked_relays,
                                                                     forbidden_sinked_relays,
                                                                     max_hops_number, number_of_hops)
    if found_forbidden:
        forbidden = True
    return chosen_node, forbidden, forbidden_sinked_relays, extra_hops


def intelligent_forbidden(sink, sinked_relays, forbidden_sinked_relays, max_hops_number, number_of_hops):
    found_forbidden = False
    # ------------------------------------------------------------------------------------------------------------------
    allowed_sinked_relays = [x for x in sinked_relays if x not in forbidden_sinked_relays]
    if len(allowed_sinked_relays) != 0:
        for i in range(len(allowed_sinked_relays)):
            if number_of_hops + allowed_sinked_relays[i][1] >= max_hops_number:
                forbidden_sinked_relays.append(allowed_sinked_relays[i])
                found_forbidden = True

    return forbidden_sinked_relays, found_forbidden
