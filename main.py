import json
from math import sqrt
import os
import time
from PersonalModules.Genetic import genetic_algorithm
from PersonalModules.VND import Variable_Neighborhood_Descent

from PersonalModules.utilities import display, bellman_ford, display2, get_stat, bars_chart, bars_chart2
from PersonalModules.greedy import greedy_algorithm
from PersonalModules.vns import Variable_Neighborhood_Search
from tabulate import tabulate
        

def create():
    free_slots = []

    # Create a grid
    chosen_grid = int(input("Choose your grid size: "))
    print("You chose the grid's size to be: ", chosen_grid, "*", chosen_grid)
    grid = chosen_grid * 20

    chosen_pas = int(input("Choose the mehs size:"))

    # Create a sink
    sink_location = input("\nDo you want the sink in the middle of the grid? (Yes or No) ")
    if sink_location == 'No':
        themesh = int(input("What mesh would you want to put the sink."))
        #sink_Y_Axis = int(input("Now enter the Y coordinate of the sink."))
        sinkX = calculate_X(themesh,chosen_grid,chosen_pas)
        sinkY = calculate_Y(themesh,chosen_grid,chosen_pas)
        print(f'x = {round(sinkX)}, y = {round(sinkY)}')
        sink = (round(sinkX), round(sinkY))
        
        print(f'The mesh: {themesh}, The grid size: {chosen_grid}, The chosen step: {chosen_pas}')
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

    custom_range = int(input("\nEnter the range you want. "))

    return grid, sink, sinkless_sentinels, free_slots, custom_range


def create2(grid_value):
    free_slots = []

    # Create a grid
    chosen_grid = grid_value
    print("You chose the grid's size to be: ", chosen_grid, "*", chosen_grid)
    grid = chosen_grid * 20

    chosen_pas = int(input("Choose the mesh size:"))

    # Create a sink
    sink_location = int(
        input("\nDo you want the sink in the middle of the grid? (Type 0 to choose a custom location) "))
    if sink_location == 0:
        themesh = int(input("What mesh would you want to put the sink."))
        #sink_Y_Axis = int(input("Now enter the Y coordinate of the sink."))
        sinkX = calculate_X(themesh,grid,chosen_pas)
        sinkY = calculate_Y(themesh,grid,chosen_pas)
        sink = (round(sinkX), round(sinkY))
        
        print(f'The mesh: {themesh}, The grid size: {grid}, The chosen step: {chosen_pas}')
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


def save(Solutions_Data):
    grid = str(Solutions_Data[0][1])
    filename = "Saved_" + grid + "x" + grid + "_Grid" + ".json"
    filepath = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions/" + filename
    with open(filepath, "w") as f:
        json.dump(Solutions_Data, f)


def save2(Solutions_Data):
    executions = str(Solutions_Data[0])
    grid = str(Solutions_Data[1])
    filename = "Saved_" + grid + "x" + grid + "_Grid" + "_Scenario_" + executions + ".json"
    filepath = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions/" + filename
    with open(filepath, "w") as f:
        json.dump(Solutions_Data, f)


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

'''
xi = (i mod sqrt(n)) * 20 + 10
yi = (i div sqrt(n)) * 20 + 10
'''
def calculate_X(themesh, grid, chosen_pas) -> int:
    X = (themesh % grid) * chosen_pas + chosen_pas /2
    return X

def calculate_Y(themesh, grid, chosen_pas)->int:
    Y = (themesh/grid) * chosen_pas+ chosen_pas /2
    return Y

def main():
    get_in = True
    # Create everything
    if get_in:
        grid, sink, sinkless_sentinels, free_slots, custom_range = create()
        max_hops_number = grid

    user_input = int(
        input("   Type 0 for one time VNS.\n   Type 1 for multiple times VNS.\n   Type 2 to Load a Grid.\n   Type 3 "
              "multiple times VNS with different grids.\n"))
    if user_input == 0:
        print("You chose One time VNS !")

        # Generate the initial solution using greedy algorithm
        print("\n   Starting Genetic algorithm...")
        sinked_sentinels, sinked_relays, free_slots, Finished, ERROR = greedy_algorithm(sink, sinkless_sentinels,
                                                                                      free_slots, max_hops_number + 1, custom_range)

        #sinked_sentinels, sinked_relays, free_slots, Finished, ERROR = genetic_algorithm(10, 4, sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range)
        print("   Genetic algorithm finished execution successfully !")

        # Get the performance before VNS, perform VNS then Get the performance after VNS
        print("\n   Please wait until some calculations are finished...")
        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays,
                                                              sinked_sentinels)
        performance_before, relays_before, hops_before = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, 0.5, 0.5)
        print("   Calculations are done !")

        print(f'\n Fitness BEFORE: {performance_before}')
        print(f'\n Relays BEFORE: {relays_before}')
        print(f'\n Hops BEFORE: {hops_before}')

        display(grid, sink, sinked_relays, sinked_sentinels, title='Greedy')

        print("\n   Starting Variable Neighborhood Search algorithm...")
        sinked_relays, free_slots = Variable_Neighborhood_Descent(grid, sink, sinked_sentinels, sinked_relays,
                                                                 free_slots, custom_range)
        print("   Variable Neighborhood Search algorithm finished execution successfully !")

        print("\n   Please wait until some calculations are finished...")
        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
        performance_after, relays_after, hops_after = get_stat(sinked_relays, sentinel_bman, cal_bman)
        print("   Calculations are done !")

        print("\n   Fitness BEFORE: ", performance_before, "\n   Fitness AFTER: ", performance_after)
        print("\n   Relays BEFORE: ", relays_before, "\n   Relays AFTER: ", relays_after)
        print("\n   Hops AVG BEFORE: ", hops_before, "\n   Hops AVG AFTER: ", hops_after)
        display(grid, sink, sinked_relays, sinked_sentinels)
    elif user_input == 1:
        Solutions_Data = []
        executions = 1
        original_free_slots = free_slots[:]

        print("You chose Multiple times VNS !\n")
        user_input = int(input("How many VNS executions you want to perform?"))

        while executions <= user_input:
            print("\n # This is the ", get_ordinal_number(executions), " VNS execution.")

            start_time = time.time()

            # Generate the initial solution using greedy algorithm
            print("\n   Starting Genetic algorithm...")
            sinked_sentinels, sinked_relays, free_slots, Finished, ERROR = greedy_algorithm(sink, sinkless_sentinels,
                                                                                            original_free_slots,
                                                                                            max_hops_number + 1, custom_range)

            #sinked_sentinels, sinked_relays, free_slots, Finished, ERROR = genetic_algorithm(100, 10, sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range)
            Greedy_grid_data = [grid, sink, sinked_relays, sinked_sentinels]
            print("   Genetic algorithm finished execution successfully !")

            # Get the performance before VNS, perform VNS then Get the performance after VNS
            print("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays,
                                                                  sinked_sentinels)
            performance_before, relays_before, hops_before = get_stat(sinked_relays, sentinel_bman, cal_bman)
            print("   Calculations are done !")

            print("\n   Starting Variable Neighborhood Search algorithm...")
            sinked_relays, free_slots = Variable_Neighborhood_Search(grid, sink, sinked_sentinels, sinked_relays,
                                                                     free_slots, custom_range)
            VNS_grid_data = [grid, sink, sinked_relays, sinked_sentinels]
            print("   Variable Neighborhood Search algorithm finished execution successfully !")

            print("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays,
                                                                  sinked_sentinels)
            performance_after, relays_after, hops_after = get_stat(sinked_relays, sentinel_bman, cal_bman)
            print("   Calculations are done !")

            print("\n   Fitness BEFORE: ", performance_before, "\n   Fitness AFTER: ", performance_after)
            print("\n   Relays BEFORE: ", relays_before, "\n   Relays AFTER: ", relays_after)
            print("\n   Hops AVG BEFORE: ", hops_before, "\n   Hops AVG AFTER: ", hops_after)

            end_time = time.time()

            # GET TIME
            total_time = int(end_time - start_time)
            hours, remainder = divmod(total_time, 3600)
            minutes, remainder = divmod(remainder, 60)
            time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

            # GET DATA
            Grid_data = [Greedy_grid_data, VNS_grid_data]
            Solutions_Data.append(
                [executions, int(grid / 20), time_string, performance_before, performance_after, relays_before,
                 relays_after,
                 hops_before, hops_after, Grid_data])
            save_data = [executions, int(grid / 20), time_string, performance_before, performance_after, relays_before,
                         relays_after, hops_before, hops_after, Grid_data]
            save2(save_data)
            executions = executions + 1

        # Aftermath
        stop = 0

        # Saving
        try:
            save(Solutions_Data)
            print("Grids data SAVED automatically !")
        except:
            print("There has been an error trying to save the grid.")

        while stop == 0:
            user_input = int(input("\nENTER:\n   1 to display the grid data table.\n   2 to STOP the program.\n   3 "
                                   "to save the grids.\n   4 Display a saved grid."))
            if user_input == 1:
                Solutions = []
                for i in range(len(Solutions_Data)):
                    Solutions.append(
                        [Solutions_Data[i][0], Solutions_Data[i][1], Solutions_Data[i][2], Solutions_Data[i][3],
                         Solutions_Data[i][4], Solutions_Data[i][5], Solutions_Data[i][6], Solutions_Data[i][7],
                         Solutions_Data[i][8]])
                headers_var = ["Executions", "Grid", "Time spent", "Average time spent",
                               "Greedy based initial solution", "VNS solution",
                               "Greedy based total relays", "VNS total relays", "Greedy Average hops",
                               "VNS Average hops"]
                print(tabulate(Solutions, headers=headers_var, showindex=False, tablefmt="rounded_outline"))
                tabulate
            elif user_input == 2:
                stop = 2
                print("\n           PROGRAM STOPPED !")
            elif user_input == 3:
                try:
                    save(Solutions_Data)
                    print("Grids data SAVED !")
                except:
                    print("There has been an error trying to save the grid.")
            elif user_input == 4:
                user_input = int(input("Type The executions number to display it's grid."))
                user_input2 = int(input("Show Greedy (Type 0) or VNS (Type 1) ?"))
                does_num_executions_exist = False
                for i in range(len(Solutions_Data)):
                    if user_input == Solutions_Data[i][0]:
                        Data = Solutions_Data[i][9][user_input2]
                        display(Data[0], Data[1], Data[2], Data[3])
                        does_num_executions_exist = True
                if not does_num_executions_exist:
                    print("\n   /!\ INVALID INPUT OR NUMBER OF HOPS DOESN'T EXIST PLEASE TRY AGAIN /!\ ")
    elif user_input == 2:
        folder_path = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions"
        files = os.listdir(folder_path)
        i = 0
        for file in files:
            i += 1
            print("\nFile ", i, ": ", file)

        user_input = str(input("\nPlease type the file name you want to load:"))

        filepath = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions/" + user_input

        # Load the variable from the file
        with open(filepath, "r") as f:
            Solutions_Data = json.load(f)

        stop = 0
        while stop == 0:
            user_input = int(input(
                "\nENTER:\n   1 to display the grid data table.\n   2 to STOP the program.\n   3 Display a saved grid."))
            if user_input == 1:
                Solutions = []
                for i in range(len(Solutions_Data)):
                    Solutions.append(
                        [Solutions_Data[i][0], Solutions_Data[i][1], Solutions_Data[i][2], Solutions_Data[i][3],
                         Solutions_Data[i][4], Solutions_Data[i][5], Solutions_Data[i][6], Solutions_Data[i][7],
                         Solutions_Data[i][8]])
                headers_var = ["Executions", "Grid", "Time spent", "Greedy performance", "VNS performance",
                               "Greedy total relays", "VNS total relays", "Greedy Average hops", "VNS Average hops"]
                print(tabulate(Solutions, headers=headers_var, showindex=False, tablefmt="rounded_outline"))

            elif user_input == 2:
                stop = 2
                print("\n           PROGRAM STOPPED !")
            elif user_input == 3:
                user_input = int(input("Type The executions number to display it's grid."))
                user_input2 = int(input("Show Greedy (Type 0) or VNS (Type 1) ?"))
                does_num_executions_exist = False
                for i in range(len(Solutions_Data)):
                    if user_input == Solutions_Data[i][0]:
                        Data = Solutions_Data[i][9][user_input2]
                        display(Data[0], Data[1], Data[2], Data[3])
                        does_num_executions_exist = True
                if not does_num_executions_exist:
                    print("\n   /!\ INVALID INPUT OR NUMBER OF HOPS DOESN'T EXIST PLEASE TRY AGAIN /!\ ")
    elif user_input == 3:
        Solutions_Data = []
        executions = 1

        print("You chose Multiple times VNS with different grids !\n")
        grid_value = [15, 20, 30, 40, 50]
        n = 0

        while executions <= 5:
            print("\n # This is the ", get_ordinal_number(executions), " VNS grid execution.")

            # Create everything
            grid, sink, sinkless_sentinels, free_slots = create2(grid_value[n])
            n = n + 1
            max_hops_number = grid

            start_time = time.time()

            # Generate the initial solution using greedy algorithm
            print("\n   Starting Genetic algorithm...")
            #sinked_sentinels, sinked_relays, free_slots, Finished, ERROR = greedy_algorithm(sink, sinkless_sentinels,
            #                                                                                free_slots,
            #                                                                                max_hops_number, custom_range)

            sinked_sentinels, sinked_relays, free_slots, Finished, ERROR = genetic_algorithm(100, 10, sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range)
            Greedy_grid_data = [grid, sink, sinked_relays, sinked_sentinels]
            print("   Genetic algorithm finished execution successfully !")

            # Get the performance before VNS, perform VNS then Get the performance after VNS
            print("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays,
                                                                  sinked_sentinels)
            performance_before, relays_before, hops_before = get_stat(sinked_relays, sentinel_bman, cal_bman)
            print("   Calculations are done !")

            print("\n   Starting Variable Neighborhood Search algorithm...")
            sinked_relays, free_slots = Variable_Neighborhood_Search(grid, sink, sinked_sentinels, sinked_relays,
                                                                     free_slots, custom_range)
            VNS_grid_data = [grid, sink, sinked_relays, sinked_sentinels]
            print("   Variable Neighborhood Search algorithm finished execution successfully !")

            print("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays,
                                                                  sinked_sentinels)
            performance_after, relays_after, hops_after = get_stat(sinked_relays, sentinel_bman, cal_bman)
            print("   Calculations are done !")

            print("\n   Fitness BEFORE: ", performance_before, "\n   Fitness AFTER: ", performance_after)
            print("\n   Relays BEFORE: ", relays_before, "\n   Relays AFTER: ", relays_after)
            print("\n   Hops AVG BEFORE: ", hops_before, "\n   Hops AVG AFTER: ", hops_after)

            end_time = time.time()

            # GET TIME
            total_time = int(end_time - start_time)
            hours, remainder = divmod(total_time, 3600)
            minutes, remainder = divmod(remainder, 60)
            time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

            # GET DATA
            Grid_data = [Greedy_grid_data, VNS_grid_data]
            Solutions_Data.append(
                [executions, int(grid / 20), time_string, performance_before, performance_after, relays_before,
                 relays_after,
                 hops_before, hops_after, Grid_data, total_time])
            executions = executions + 1

        # Aftermath
        stop = 0
        avg_time = 0
        for i in range(len(Solutions_Data)):
            avg_time = avg_time + Solutions_Data[i][10]

        # Get time
        days, remainder = divmod(avg_time, 86400)
        hours, remainder = divmod(avg_time, 3600)
        minutes, remainder = divmod(avg_time, 60)
        avg_time_string = f"{days:02.0f}D_{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"
        while stop == 0:
            user_input = int(input("\nENTER:\n   1 to display the grid data table.\n   2 to STOP the program.\n   3 "
                                   "to save the grids.\n   4 Display a saved grid.\n   5 Display bars chart."))
            if user_input == 1:
                Solutions = []
                for i in range(len(Solutions_Data)):
                    Solutions.append(
                        [Solutions_Data[i][0], Solutions_Data[i][1], avg_time_string, Solutions_Data[i][2],
                         Solutions_Data[i][3],
                         Solutions_Data[i][4], Solutions_Data[i][5], Solutions_Data[i][6], Solutions_Data[i][7],
                         Solutions_Data[i][8]])
                headers_var = ["Scenarios", "Grids", "Average time", "Time spent",
                               "Initial fitness", "VNS fitness",
                               "Initial total relays", "VNS total relays", "Initial Average hops",
                               "VNS Average hops"]
                print(tabulate(Solutions, headers=headers_var, showindex=False, tablefmt="rounded_outline"))
            elif user_input == 2:
                stop = 2
                print("\n           PROGRAM STOPPED !")
            elif user_input == 3:
                try:
                    save(Solutions_Data)
                    print("Grids data SAVED !")
                except:
                    print("There has been an error trying to save the grid.")
            elif user_input == 4:
                user_input = int(input("Type The executions number to display it's grid."))
                user_input2 = int(input("Show Greedy (Type 0) or VNS (Type 1) ?"))
                does_num_executions_exist = False
                for i in range(len(Solutions_Data)):
                    if user_input == Solutions_Data[i][0]:
                        Data = Solutions_Data[i][9][user_input2]
                        display(Data[0], Data[1], Data[2], Data[3])
                        does_num_executions_exist = True
                if not does_num_executions_exist:
                    print("\n   /!\ INVALID INPUT OR NUMBER OF HOPS DOESN'T EXIST PLEASE TRY AGAIN /!\ ")
            elif user_input == 5:
                initial_total_relays = []
                vns_total_relays = []
                initial_average_hops = []
                vns_average_hops = []
                for i in range(len(Solutions_Data)):
                    initial_total_relays.append(Solutions_Data[i][5])
                    vns_total_relays.append(Solutions_Data[i][6])
                    initial_average_hops.append(Solutions_Data[i][7])
                    vns_average_hops.append(Solutions_Data[i][8])
                values = [initial_total_relays, vns_total_relays, initial_average_hops, vns_average_hops]
                bars_chart(values)
    elif user_input == 4:
        folder_path = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions"
        files = os.listdir(folder_path)
        i = 0
        for file in files:
            i += 1
            print("\nFile ", i, ": ", file)

        user_input = str(input("\nPlease type the file name you want to load:"))

        filepath = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions/" + user_input

        # Load the variable from the file
        with open(filepath, "r") as f:
            Solutions_Data = json.load(f)

        stop = 0
        avg_time = 0
        for i in range(len(Solutions_Data)):
            avg_time = avg_time + Solutions_Data[i][10]

        # Get time
        days, remainder = divmod(avg_time, 86400)
        avg_time = avg_time - 86400
        hours, remainder = divmod(avg_time, 3600)
        minutes, remainder = divmod(avg_time, 60)
        avg_time_string = f"{days:02.0f}D_{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"
        while stop == 0:
            user_input = int(input(
                "\nENTER:\n   1 to display the grid data table.\n   2 to STOP the program.\n   3 Display a saved "
                "grid.\n   4 Display bars chart."))
            if user_input == 1:
                Solutions = []
                for i in range(len(Solutions_Data)):
                    Solutions.append(
                        [Solutions_Data[i][0], Solutions_Data[i][1], avg_time_string, Solutions_Data[i][2],
                         Solutions_Data[i][3],
                         Solutions_Data[i][4], Solutions_Data[i][5], Solutions_Data[i][6], Solutions_Data[i][7],
                         Solutions_Data[i][8]])
                headers_var = ["Scenarios", "Grids", "Average time", "Time spent",
                               "Initial fitness", "VNS fitness",
                               "Initial total relays", "VNS total relays", "Initial Average hops",
                               "VNS Average hops"]
                print(tabulate(Solutions, headers=headers_var, showindex=False, tablefmt="rounded_outline"))

            elif user_input == 2:
                stop = 2
                print("\n           PROGRAM STOPPED !")
            elif user_input == 3:
                user_input = int(input("Type The executions number to display it's grid."))
                user_input2 = int(input("Show Greedy (Type 0) or VNS (Type 1) ?"))
                does_num_executions_exist = False
                for i in range(len(Solutions_Data)):
                    if user_input == Solutions_Data[i][0]:
                        Data = Solutions_Data[i][9][user_input2]
                        display(Data[0], Data[1], Data[2], Data[3])
                        does_num_executions_exist = True
                if not does_num_executions_exist:
                    print("\n   /!\ INVALID INPUT OR NUMBER OF HOPS DOESN'T EXIST PLEASE TRY AGAIN /!\ ")
            elif user_input == 4:
                initial_total_relays = []
                vns_total_relays = []
                initial_average_hops = []
                vns_average_hops = []
                timee = []
                for i in range(len(Solutions_Data)):
                    initial_total_relays.append(Solutions_Data[i][5])
                    vns_total_relays.append(Solutions_Data[i][6])
                    initial_average_hops.append(Solutions_Data[i][7])
                    vns_average_hops.append(Solutions_Data[i][8])
                    timee.append(Solutions_Data[i][10])
                values = [initial_total_relays, vns_total_relays, initial_average_hops, vns_average_hops]
                bars_chart(values)
                bars_chart2(timee)


if __name__ == '__main__':
    main()
