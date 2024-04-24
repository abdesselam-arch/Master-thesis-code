'''
    Upper Bound Confident function (UCB1)

    Author: Ahmed Nour ABDESSELAM
    Date: March, 2024

    VNS-UCB1 Algorithm:
    A hybrid approach that combines Variable Neighborhood Descent (VND) with the Upper Confidence Bound (UCB1) algorithm.
    1. VND explores the solution space by iteratively applying a sequence of neighborhood structures.
    2. UCB1 dynamically selects the neighborhood structure to explore at each iteration based on estimated rewards.
    3. By combining exploration (VND) and exploitation (UCB1), the algorithm efficiently searches for promising solutions.
    4. Rewards or regrets are calculated based on the performance of selected neighborhoods, guiding the search process.
    5. The algorithm aims to find high-quality solutions while balancing between exploration of new regions and exploitation of known promising areas.
'''
import math
import random

from matplotlib import pyplot as plt
import numpy as np

'''
Visualizing the results
'''
def Credit_Assignment(improvement, previous, after, l):
    '''
    Calculate credit (regret or reward) based on improvement.

    Args:
        improvement (bool): Wheter the action resulted in inmprovement.
        previous (int): The fitness before neighborhood action.
        after (int): The fitness after neighborhood action.

    Retruns:
        reward (int): Non binary reward
    '''
    # Rs is the distance between the prior solution and current solution
    Rs = abs(previous - after)

    if improvement:
        if l == 0 or l == 4:
            print(f'The reward {l}: {20 * Rs}')
            return 20 * Rs 
        elif l == 1:
            print(f'The reward {l}: {20 * Rs}')
            return 40 * Rs
        elif l == 2:
            print(f'The reward {l}: {20 * Rs}')
            return 30 * Rs
        elif l == 3:
            print(f'The reward {l}: {20 * Rs}')
            return 25 * Rs
    else:
        Penalty = -(Rs * 30)
        return Penalty

# UCB1 policy implementation ---------------------------------------------------------------------------------------------------------------------------
def UCB1_policy(lmax, qualities, exploration_factor, total_actions):
    """
    Applies UCB1 policy to generate neighborhood recommendations.

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
            '''
            At first, action_count = 0 meanning no neighborhood has been called
            we can't base our calculation on that, we can randomize the action chosen (GVNS Shaking)
            we can also call each action sequentially, 1->2->3->4->5
            '''
            return random.randint(1, 5)
        else:                        
            exploration_term = exploration_factor * math.sqrt((2 * math.log(total_action_count)) / (action_count + 1))
            return quality + exploration_term
    
    action_counts = [0] * lmax

    # Choose neighborhood using UCB1
    ucb_scores = [calculate_score(action_counts[i], total_actions, qualities[i]) for i in range(lmax)]
    chosen_neighborhood = np.argmax(ucb_scores)
    print(f'UCB1 Neighborhood chosen: {chosen_neighborhood+1}')

    if chosen_neighborhood == 0:
        print('N1(s) - Add random relay chosen\n')
    elif chosen_neighborhood == 1:
        print('N2(s) - Delete random relay chosen\n')
    elif chosen_neighborhood == 2:
        print('N3(s) - Relay next to sentinel deleted chosen\n')
    elif chosen_neighborhood == 3:
        print('N4(s) - Relays relocated chosen\n')
    elif chosen_neighborhood == 4:
        print('N5(s) - Relay added next to sentinel with no neighbors chosen\n')

    # Update action counts and neighborhood qualities based on the outcome of the chosen action
    action_counts[chosen_neighborhood] += 1
    
    return chosen_neighborhood, action_counts[chosen_neighborhood]
