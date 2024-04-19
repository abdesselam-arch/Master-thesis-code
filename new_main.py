import math
import time

from Home import get_Diameter
from PersonalModules.Genetic import genetic_algorithm
from PersonalModules.UCB_VND import UCB_VND
from PersonalModules.utilities import bellman_ford, get_stat, len_sinked_relays


def create():
    free_slots = []

    # Create a grid
    chosen_grid = int(input("Choose your grid size: "))
    print("You chose the grid's size to be: ", chosen_grid, "*", chosen_grid)
    grid = chosen_grid * 20

    # Create a sink
    sink_location = int(
        input("\nDo you want the sink in the middle of the grid? (Type 0 to choose a custom location) "))
    if sink_location == 0:
        sink_X_Axis = int(input("Enter the X coordinate of the sink."))
        sink_Y_Axis = int(input("Now enter the Y coordinate of the sink."))
        sink = (sink_X_Axis, sink_Y_Axis)
    else:
        if (chosen_grid % 2) == 0:
            sink = ((grid / 2) + 10, (grid / 2) - 10)
        elif (chosen_grid % 2) == 1:
            sink = (((grid - 20) / 2) + 10, ((grid - 20) / 2) + 10)

    # Create sentinels
    sinkless_sentinels = [(x, 10) for x in range(10, grid + 10, 20)] + \
                         [(x, grid - 10) for x in range(10, grid + 10, 20)] + \
                         [(10, y) for y in range(30, grid - 10, 20)] + \
                         [(grid - 10, y) for y in range(30, grid - 10, 20)]

    # Create the free slots
    for x in range(30, grid - 10, 20):
        for y in range(30, grid - 10, 20):
            if sink != (x, y):
                free_slots.append((x, y))
    return grid, sink, sinkless_sentinels, free_slots

def get_ordinal_number(n):
    if n % 100 in [11, 12, 13]:
        suffix = "th"
    else:
        last_digit = n % 10
        if last_digit == 1:
            suffix = "st"
        elif last_digit == 2:
            suffix = "nd"
        elif last_digit == 3:
            suffix = "rd"
        else:
            suffix = "th"
    return str(n) + suffix

def isPrime(x):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    tf = eng.isprime(x)
    return tf

def main():
    get_in = True
    # Create everything
    if get_in:
        grid, sink, sinkless_sentinels, free_slots = create()
        max_hops_number = grid

    user_input = int(
        input("     Type 1 for multiple times VNS.\n"))
    

    if user_input == 1:
        executions = 1
        vns_avg_hops = 0
        vns_avg_relays = 0
        vns_avg_performance = 0
        vns_avg_diameter = 0
        ga_avg_hops = 0
        ga_avg_relays = 0
        ga_avg_performance = 0
        ga_avg_diameter = 0

        print("You chose Multiple times Greedy !\n")
        user_input = int(input("How many Greedy executions you want to perform?"))

        simulation_start_time = time.time()
        execution_times = []
        while executions <= user_input:
            print("\n # This is the ", get_ordinal_number(executions), " Genetic + UCB_VND execution.")

            start_time = time.time()
            # Generate the initial solution using greedy algorithm
            print("\n   Starting Genetic algorithm...")
            genetic_sinked_sentinels, genetic_sinked_relays, genetic_free_slots, Finished, ERROR = genetic_algorithm(3, 10, sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range = 30, mesh_size = 20)

    
            print("   Greedy algorithm finished execution successfully !")

            # Get the performance before VNS, perform VNS then Get the performance after VNS
            print("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, genetic_cal_bman = bellman_ford(grid, genetic_free_slots, sink, genetic_sinked_relays,
                                                                    genetic_sinked_sentinels)
            performance_before, relays_before, hops_before = get_stat(genetic_sinked_relays, sentinel_bman, genetic_cal_bman, grid, genetic_free_slots, sink, genetic_sinked_sentinels, mesh_size = 20, alpha = 0.5, beta = 0.5)
            diameter_before = get_Diameter(sentinel_bman, genetic_cal_bman, mesh_size = 20)
            print("   Calculations are done !")

            sinked_relays = genetic_sinked_relays
            sinked_sentinels = genetic_sinked_sentinels
            free_slots = genetic_free_slots

            ga_diameter = diameter_before
            ga_avg_hops += hops_before
            ga_avg_relays += relays_before
            ga_avg_performance += performance_before
            ga_avg_diameter += ga_diameter

            print('Starting the UCB_VND algorithm now!!')

            sinked_relays, free_slots = UCB_VND(grid, sink, sinked_sentinels, sinked_relays,
                                                                        free_slots, custom_range = 30, mesh_size = 20, lmax=5, alpha= 0.5, beta=0.5)
            print("   Upper Confidence Bounde + Variable Neighborhood Descent algorithm finished execution successfully !")

            print("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
            
            performance_after, relays_after, hops_after = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, mesh_size = 20, alpha = 0.5, beta = 0.5)
            
            diameter_after = get_Diameter(sentinel_bman, cal_bman, mesh_size = 20)
            relays_after = len_sinked_relays(sinked_relays)
            print("   Calculations are done !")

            print(f"\nFitness BEFORE: {performance_before}")
            print(f"Fitness AFTER: {performance_after}\n")
                
            print(f"Relays BEFORE: {relays_before}")
            print(f"Relays AFTER: {relays_after}\n")

            print(f"Network diameter BEFORE: {diameter_before}")
            print(f"Network diameter AFTER: {diameter_after}\n")

            print(f"Hops Average BEFORE: {hops_before}")
            print(f"Hops Average AFTER: {hops_after}\n")

            vns_avg_hops += hops_after
            vns_avg_relays += relays_after
            vns_avg_performance += performance_after
            vns_avg_diameter += diameter_after

            executions = executions + 1

            end_time = time.time()
            # GET TIME
            total_time = int(end_time - start_time)
            execution_times.append(total_time)
            hours, remainder = divmod(total_time, 3600)
            minutes, remainder = divmod(remainder, 60)
            time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

            print(f'GA UCB_VND Execution time: {time_string}')

        simulation_end_time = time.time()
        # GET TIME
        total_time = int(simulation_end_time - simulation_start_time)
        hours, remainder = divmod(total_time, 3600)
        minutes, remainder = divmod(remainder, 60)
        simulation_time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

        total_execution_time = sum(execution_times)
        #   Calculate average execution time
        average_execution_time = total_execution_time / len(execution_times)
        hours, remainder = divmod(average_execution_time, 3600)
        minutes, remainder = divmod(remainder, 60)
        avg_time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

        print('\n\nSimulation Results:\n')

        print('GA Results AVERAGE:')

        print(f'Relays AVERAGE: {math.ceil(ga_avg_relays / user_input)}')
        print(f'Hops AVERAGE: {math.ceil(ga_avg_hops / user_input)}')
        print(f'Performance AVERAGE: {ga_avg_performance / user_input}')
        print(f'Diameter AVERAGE: {math.ceil(ga_avg_diameter / user_input)}')
            
        print('\nUCB_VND Results AVERAGE:')

        print(f'Relays AVERAGE: {math.ceil(vns_avg_relays / user_input)}')
        print(f'Hops AVERAGE: {math.ceil(vns_avg_hops / user_input)}')
        print(f'Performance AVERAGE: {vns_avg_performance / user_input}')
        print(f'Diameter AVERAGE: {math.ceil(vns_avg_diameter / user_input)}')

        avg_execution_time = total_time / user_input
        avg_hours, avg_remainder = divmod(avg_execution_time, 3600)
        avg_minutes, avg_remainder = divmod(avg_remainder, 60)
        avg_time_string = f"{avg_hours:02.0f}H_{avg_minutes:02.0f}M_{avg_remainder:02.0f}       "

        print(f'\nExecution time AVERAGE: {avg_time_string}')
        print(f'Total execution time: {time_string}')
   


if __name__ == '__main__':
    main()