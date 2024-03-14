'''
    Upper Bound Confident function
'''
import math
import random

import numpy as np
from PersonalModules.utilities import bellman_ford, epsilon_constraints

def Credit(improvement):
    '''
    Calculate credit (regret or reward) based on improvement.

    Args:
        improvement (bool): Wheter the action resulted in inmprovement.

    Retruns:
        reward (int): Non binary reward scheme
    '''
    if improvement:
        Reward = 1
        return Reward
    else:
        Regret = 0
        return Regret

def UCB1(grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, lmax, alpha, beta, exploration_factor):
        """
        Choose a neighborhood dynamically using the UCB1 algorithm.

        Args:
            lmax (int): Maximum neighborhood.
            exploration_factor (float): Exploration factor for UCB1.

        Returns:
            chosen_neighborhood (int): Chosen neighborhood.
        """    
        def calculate_score(action_count, total_action_count, quality):
            """
            Calculate the UCB1 score for a neighborhood.

            Args:
                action_count (int): Number of times the neighborhood has been chosen.
                total_action_count (int): Total number of times any neighborhood has been chosen.
                quality (float): Quality of the neighborhood (e.g., fitness improvement).

            Returns:
                score (float): UCB1 score for the neighborhood.
            """
            if action_count == 0:
                return random.randint(1, 5)
            else:        
                distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
                quality = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size, alpha, beta)
                
                exploration_term = exploration_factor * math.sqrt((2 * math.log(total_action_count)) / action_count)
                return quality + exploration_term
    
        total_actions = 0
        action_counts = [0] * lmax
        qualities = [0.0] * lmax

        while total_actions < lmax:
            total_actions += 1

            # Choose neighborhood using UCB1
            ucb_scores = [calculate_score(action_counts[i], total_actions, qualities[i]) for i in range(lmax)]
            chosen_neighborhood = np.argmax(ucb_scores)

            if chosen_neighborhood == 1:
                print('N1(s) - Add random relay chosen ')
            
            elif chosen_neighborhood == 2:
                print('N2(s) - Delete random relay chosen ')
            
            elif chosen_neighborhood == 3:
                print('N3(s) - Relay next to sentinel deleted chosen')
            
            elif chosen_neighborhood == 4:
                print('N4(s) - Relays swaped')
            
            elif chosen_neighborhood == 5:
                print('N5(s) - Relay added next to sentinel with no neighbors chosen')

            # Update action counts and neighborhood qualities based on the outcome of the chosen action
            action_counts[chosen_neighborhood] += 1
            # Update qualities[chosen_neighborhood] based on fitness improvement or other metrics

        return chosen_neighborhood
