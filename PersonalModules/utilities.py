from heapq import heappop, heappush
import heapq
import json
import math
from multiprocessing import Process, Queue
import os
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.animation import FuncAnimation
import matlab.engine
import psutil

def get_Diameter(sentinel_bman, cal_bman, mesh_size):
    sentinel_relays = sentinel_relay(sentinel_bman)
    if 999 in sentinel_bman:
        return cal_bman/mesh_size
    else:
        return max(sentinel_relays)

def display(grid, sink, sinked_relays, sentinels, title):
    # Create a new plot
    fig, ax = plt.subplots()

    # Set the axis limits and grid size
    ax.set_xlim(0, grid)
    ax.set_ylim(0, grid)

    # Add gridlines to the plot
    ax.grid(True)

    # Plot the points as circles
    for i in range(len(sinked_relays)):
        ax.plot(sinked_relays[i][0][0], sinked_relays[i][0][1], marker='o', color='black')
    for i in range(len(sentinels)):
        ax.plot(sentinels[i][0], sentinels[i][1], marker='s', color='blue')
    ax.plot(sink[0], sink[1], marker='^', color='red')

    ax.set_title(title)

    # Show the plot
    ticks = []
    for i in range(0, grid + 1, 20):
        ticks.append(i)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.show()

def display2(grid, sink, sinked_relays, sentinels):
    # Create a new plot
    fig, ax = plt.subplots()

    # Set the axis limits and grid size
    ax.set_xlim(0, grid)
    ax.set_ylim(0, grid)

    # Add gridlines to the plot
    ax.grid(True)

    # Plot the points as circles
    for relay in sinked_relays:
        ax.plot(relay[0], relay[1], marker='o', color='black')

    for sentinel in sentinels:
        ax.plot(sentinel[0], sentinel[1], marker='s', color='orange')

    ax.plot(sink[0], sink[1], marker='^', color='red')

    # Show the plot
    ticks = []
    for i in range(0, grid + 1, 20):
        ticks.append(i)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.show()

def display_realtime(grid, sink, sinked_relays, sentinels):
    # create a figure and axis for the plot
    fig, ax = plt.subplots()

    # initialize an empty plot
    line, = ax.plot([], [])

    # Add gridlines to the plot
    ax.grid(True)

    # Show the plot
    ticks = []
    for i in range(0, grid + 1, 20):
        ticks.append(i)
    plt.xticks(ticks)
    plt.yticks(ticks)

    # define a function that updates the plot with new data
    def update_plot(frame):
        # generate some random data
        # Plot the points as circles
        for i in range(len(sinked_relays)):
            ax.plot(sinked_relays[i][0][0], sinked_relays[i][0][1], marker='o', color='black')
        for i in range(len(sentinels)):
            ax.plot(sentinels[i][0], sentinels[i][1], marker='s', color='orange')
        ax.plot(sink[0], sink[1], marker='^', color='red')

        # Set the axis limits and grid size
        ax.set_xlim(0, grid)
        ax.set_ylim(0, grid)

        return line,

    # create an animation that calls the update_plot function every 1000 milliseconds
    ani = FuncAnimation(fig, update_plot, interval=1000)

    # show the plot
    plt.show()

def bars_chart(values):
    import matplotlib.pyplot as plt
    import numpy as np

    # Data for the bar chart
    groups = ['Initial total relays', 'VNS total relays', 'Initial average hops', 'VNS average hops']
    categories = ['15x15', '20x20', '30x30', '40x40', '50x50']
    colors = ['#1A3880', '#3371FF', '#9F2020', '#FF3333']

    # Set the figure size
    # plt.figure(figsize=(100, 60))

    # Set the width of each bar
    bar_width = 0.15

    # Set the x coordinate of each bar
    x = np.arange(len(categories))

    # Create a bar for each group and category
    for i, group in enumerate(groups):
        plt.bar(x + (i * bar_width), values[i], width=bar_width, label=group, color=colors[i])
        for j, value in enumerate(values[i]):
            plt.text(x[j] + (i * bar_width), value + 0.2, str(round(value, 1)), ha='center')

    # Set the x-axis tick labels
    plt.xticks(x + bar_width * (len(groups) - 1) / 2, categories)

    # Add labels and title
    plt.xlabel('Grids')
    plt.ylabel('Values')
    plt.title('Bar Chart with 4 Groups and 5 Different Grids')

    # Add a legend
    plt.legend()

    '''# Add a legend with padding
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')

    # Adjust the layout to prevent overlap
    plt.tight_layout()'''

    '''fig_manager = plt.get_current_fig_manager()
    fig_manager.full_screen_toggle()
    fig_manager.window.state('zoomed')'''
    plt.show()

def bars_chart2(values):
    import matplotlib.pyplot as plt

    # Sample data
    categories = ['15x15', '20x20', '30x30', '40x40', '50x50']

    # Plotting the bar chart
    plt.bar(categories, values, color='#132745')

    # Adding labels and title
    plt.xlabel('Grids')
    plt.ylabel('Time (in seconds)')
    plt.title('Bars Chart of different Grids in function of time.')

    for i, v in enumerate(values):
        tim = v
        hours, remainder = divmod(tim, 3600)
        minutes, remainder = divmod(tim, 60)
        avg_time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"
        plt.text(i, v, avg_time_string, ha='center', va='bottom')
    # Displaying the chart
    plt.show()

def plot_fitness_comparison():
    # Data for the grids and corresponding fitness values
    grid_sizes = ['17x17', '20x20', '25x25', '30x30', '35x35']
    fitness_ga = [111.25, 156.1, 254.5, 377, 522.1]
    #fitness_ucb_gvns = [56, 73, 133.9, 171.95, 222]
    fitness_ucb_gvns = [62, 83, 134.5, 175.5, 226.05]

    # Setting up the bar chart
    bar_width = 0.35
    index = range(len(grid_sizes))

    fig, ax = plt.subplots()

    # Plotting the bars for GA and UCB_GVNS
    bars1 = ax.bar(index, fitness_ga, bar_width, label='GA', color='#3c6aa6')
    bars2 = ax.bar([i + bar_width for i in index], fitness_ucb_gvns, bar_width, label='UCB_GVNS', color='green')

    # Adding labels and titles
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Comparison between GA and UCB_GVNS')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(grid_sizes)
    ax.legend()

    # Display the plot
    plt.show()

def test(grid, sink, sinked_relays, sentinels):
    First_time = True
    # ----------------------------------------------------------------------------------------------------------
    # create a figure and axis for the plot
    fig, ax = plt.subplots()

    # initialize an empty plot
    line, = ax.plot([], [])

    # Add gridlines to the plot
    ax.grid(True)

    # Show the plot
    ticks = []
    for i in range(0, grid + 1, 20):
        ticks.append(i)
    plt.xticks(ticks)
    plt.yticks(ticks)

    # define a function that updates the plot with new data
    def update_plot(frame):
        # generate some random data
        # Plot the points as circles
        for i in range(len(sinked_relays)):
            ax.plot(sinked_relays[i][0][0], sinked_relays[i][0][1], marker='o', color='black')
        for i in range(len(sentinels)):
            ax.plot(sentinels[i][0], sentinels[i][1], marker='s', color='orange')
        ax.plot(sink[0], sink[1], marker='^', color='red')

        # Set the axis limits and grid size
        ax.set_xlim(0, grid)
        ax.set_ylim(0, grid)

        return line,

    # create an animation that calls the update_plot function every 1000 milliseconds
    ani = FuncAnimation(fig, update_plot, interval=1000)

    # show the plot
    if First_time:
        First_time = False
        plt.show()

def plot_histogram(neighborhood_counts, max_iterations):
    """
    Plot a histogram of neighborhoods selections as percentages.

    Args:
        neighborhood_counts (list): List of counts for each neighborhood.
        max_iterations (int): Total number of iterations.

    Returns:
        None
    """
    percentages = [count / max_iterations * 100 for count in neighborhood_counts]
    neighborhoods = [f'N({i+1})' for i in range(len(neighborhood_counts))]
    plt.bar(neighborhoods, percentages)
    plt.title('Histogram of Neighborhoods selections (Percentage)')
    plt.xlabel('Neighborhoods')
    plt.ylabel('Percentage of total iterations')
    plt.show()

    # Print percentages truncated to two decimal places
    total_percentage = sum(percentages)
    for i, percentage in enumerate(percentages):
        if i < len(percentages) - 1:
            print(f'Percentage of selecting neighborhood N({i}): {percentage:.2f}%')
        else:
            # For the last percentage, adjust to ensure the total is 100%
            print(f'Percentage of selecting neighborhood N({i}): {abs(100 - (total_percentage - percentage)):.2f}%')

def plot_fitness_improvement(iteration_numbers, fitness_values, neighborhoods):
    # Plot fitness values against iteration numbers
    plt.plot(iteration_numbers, fitness_values, marker='o')
    plt.title('Fitness Improvement Over Time')

    # Check if the length of neighborhoods is too long
    max_chars_per_line = 250
    neighborhoods_str = ',  '.join(map(str, neighborhoods))
    if len(neighborhoods_str) > max_chars_per_line:
        # Split neighborhoods into multiple lines
        neighborhoods_lines = [neighborhoods_str[i:i+max_chars_per_line] for i in range(0, len(neighborhoods_str), max_chars_per_line)]
        neighborhoods_label = '\n'.join(neighborhoods_lines)
        plt.xlabel('(' + neighborhoods_label + ')')
    else:
        plt.xlabel('(' + neighborhoods_str + ')')

    plt.ylabel('Fitness')
    plt.grid(True)
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks on x-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    
    plt.tight_layout()  # Adjust layout to prevent overlap of labels
    plt.show()

def track_neighborhood_sequence(prior_neighborhood, current_neighborhood, sequence_counts):
    if prior_neighborhood is not None:
        sequence = (prior_neighborhood, current_neighborhood)
        sequence_key = f"N({sequence[0] + 1}) -> N({sequence[1] + 1})"  # Convert tuple to string
        sequence_counts[sequence_key] = sequence_counts.get(sequence_key, 0) + 1

def print_sequence_counts(sequence_counts):
    print("\nNeighborhood Sequences:")
    print("======================")
    print("Sequence   |   Count")
    print("----------------------")
    for sequence, count in sequence_counts.items():
        print(f"  {sequence}      |   {count} times")

def write_sequence_counts_to_json(grid_size, sink_coordinates, num_relays_deployed, diameter, sequence_counts, fitness, filename, filepath):
    data = {
        "grid_size": grid_size,
        "sink_coordinates": sink_coordinates,
        "num_relays_deployed": num_relays_deployed,
        "diameter": diameter,
        "Fitness": fitness,
        "sequence_counts": sequence_counts
    }

    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)

    with open(os.path.join(filepath, filename), 'w') as json_file:
        json.dump(data, json_file, indent=4)

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        # Start monitoring memory and CPU usage
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
        start_cpu = process.cpu_percent()

        # Execute the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Calculate memory and CPU consumption
        end_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
        end_cpu = process.cpu_percent()

        # Print the results
        print(f"Memory consumed: {abs(end_memory - start_memory):.2f} MB")
        print(f"CPU cores consumed: {end_cpu - start_cpu:.2f}%")
        print(f"Execution time: {end_time - start_time:.2f} seconds")

        return result

    return wrapper
# --------------------------------------------------------------------------------------------------------------

def bellman_ford(grid, free_slots, sink, relays, sentinels):
    chosen_grid = int(grid / 20)
    meshes = chosen_grid * chosen_grid
    start = -1
    for i in range(meshes):
        x = (i % chosen_grid) * 20 + (20 / 2)
        y = (i // chosen_grid) * 20 + (20 / 2)
        if (x, y) == sink:
            start = i
            break

    # O(n^2) complexity for this part
    # Create adjacency matrix
    graph = [[0 for j in range(meshes)] for i in range(meshes)]
    extra_array = [sink]
    extra_array.extend(sentinels)
    for a in range(len_sinked_relays(relays)):
        extra_array.append(relays[a][0])
    for i in range(meshes):
        for j in range(meshes):
            xi = (i % chosen_grid) * 20 + (20 / 2)
            yi = (i // chosen_grid) * 20 + (20 / 2)
            xj = (j % chosen_grid) * 20 + (20 / 2)
            yj = (j // chosen_grid) * 20 + (20 / 2)
            if i != j:
                if (xi, yi) in extra_array and (xj, yj) in extra_array:
                    if (xi, yi) in sentinels and (xj, yj) in sentinels:
                        pass
                    else:
                        if math.dist((xi, yi), (xj, yj)) < 30:
                            graph[i][j] = 1

    INF = float('inf')

    # Initialize distance array with infinity for all vertices except the start vertex
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0

    # O((n-1)*n*n) => O(n^3)
    # Relax edges repeatedly
    for _ in range(n - 1):
        for u in range(n):
            for v in range(n):
                if graph[u][v] != 0:  # If there's an edge between u and v
                    if dist[u] + graph[u][v] < dist[v]:  # If a shorter path to v is found through u
                        dist[v] = dist[u] + graph[u][v]

    # Acquire distances of only sentinels and calculate the sum of them
    sentinel_bman = []
    cal_bman = 0
    # Complexity O(meshes) => O(n)
    for i in range(meshes):
        xi = (i % chosen_grid) * 20 + (20 / 2)
        yi = (i // chosen_grid) * 20 + (20 / 2)
        if (xi, yi) in sentinels:
            if dist[i] == INF:  # Check if a sentinel is unreachable
                sentinel_bman.append(999)  # Set distance to 0 if unreachable
                cal_bman += 999
            else:
                sentinel_bman.append(dist[i])
                cal_bman += dist[i]

    # total complexity is O(n^3) as the bellman_ford algorithm dictate
    return dist, sentinel_bman, cal_bman

def floyd_warshall(grid, sink, relays, sentinels):
    chosen_grid = int(grid / 20)
    meshes = chosen_grid * chosen_grid
    start = -1
    for i in range(meshes):
        x = (i % chosen_grid) * 20 + (20 / 2)
        y = (i // chosen_grid) * 20 + (20 / 2)
        if (x, y) == sink:
            start = i
            break

    # Create adjacency matrix
    graph = [[math.inf for _ in range(meshes)] for _ in range(meshes)]
    extra_array = [sink]
    extra_array.extend(sentinels)
    for a in range(len(relays)):
        extra_array.append(relays[a][0])
    for i in range(meshes):
        for j in range(meshes):
            xi = (i % chosen_grid) * 20 + (20 / 2)
            yi = (i // chosen_grid) * 20 + (20 / 2)
            xj = (j % chosen_grid) * 20 + (20 / 2)
            yj = (j // chosen_grid) * 20 + (20 / 2)
            if i != j:
                if (xi, yi) in extra_array and (xj, yj) in extra_array:
                    if (xi, yi) in sentinels and (xj, yj) in sentinels:
                        pass
                    else:
                        if math.dist((xi, yi), (xj, yj)) < 30:
                            graph[i][j] = 1

    # Apply Floyd-Warshall algorithm
    n = len(graph)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])

    # Acquire distances of only sentinels and calculate the sum of them
    sentinel_bman = []
    cal_bman = 0
    for i in range(meshes):
        xi = (i % chosen_grid) * 20 + (20 / 2)
        yi = (i // chosen_grid) * 20 + (20 / 2)
        if (xi, yi) in sentinels:
            if graph[start][i] == math.inf:  # Check if a sentinel is unreachable
                sentinel_bman.append(999)  # Set distance to 0 if unreachable
                cal_bman += 999
            else:
                sentinel_bman.append(graph[start][i])
                cal_bman += graph[start][i]

    return graph[start], sentinel_bman, cal_bman

def dijkstra(grid, sink, relays, sentinels):
    chosen_grid = int(grid / 20)
    meshes = chosen_grid * chosen_grid
    start = -1
    for i in range(meshes):
        x = (i % chosen_grid) * 20 + (20 / 2)
        y = (i // chosen_grid) * 20 + (20 / 2)
        if (x, y) == sink:
            start = i
            break

    # Create adjacency list
    graph = [[] for _ in range(meshes)]
    extra_array = [sink]
    extra_array.extend(sentinels)
    for a in range(len(relays)):
        extra_array.append(relays[a][0])
    for i in range(meshes):
        xi = (i % chosen_grid) * 20 + (20 / 2)
        yi = (i // chosen_grid) * 20 + (20 / 2)
        for j in range(meshes):
            xj = (j % chosen_grid) * 20 + (20 / 2)
            yj = (j // chosen_grid) * 20 + (20 / 2)
            if i != j:
                if (xi, yi) in extra_array and (xj, yj) in extra_array:
                    if (xi, yi) in sentinels and (xj, yj) in sentinels:
                        pass
                    else:
                        if math.dist((xi, yi), (xj, yj)) < 30:
                            graph[i].append((j, 1))  # Assuming unit weight for all edges

    INF = float('inf')

    # Initialize distance array with infinity for all vertices except the start vertex
    dist = [INF] * meshes
    dist[start] = 0

    # Priority queue to store vertices with their distances
    pq = [(0, start)]

    # Dijkstra's algorithm
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    # Acquire distances of only sentinels and calculate the sum of them
    sentinel_bman = []
    cal_bman = 0
    for i in range(meshes):
        xi = (i % chosen_grid) * 20 + (20 / 2)
        yi = (i // chosen_grid) * 20 + (20 / 2)
        if (xi, yi) in sentinels:
            if dist[i] == INF:  # Check if a sentinel is unreachable
                sentinel_bman.append(999)  # Set distance to 0 if unreachable
                cal_bman += 999
            else:
                sentinel_bman.append(dist[i])
                cal_bman += dist[i]

    return dist, sentinel_bman, cal_bman


def bellman_ford_heuristic(grid, sink, relays, sentinels):
    chosen_grid = int(grid / 20)
    meshes = chosen_grid * chosen_grid
    start = -1
    for i in range(meshes):
        x = (i % chosen_grid) * 20 + (20 / 2)
        y = (i // chosen_grid) * 20 + (20 / 2)
        if (x, y) == sink:
            start = i
            break

    # Create adjacency matrix
    graph = [[math.inf for _ in range(meshes)] for _ in range(meshes)]
    extra_array = [sink]
    extra_array.extend(sentinels)
    for a in range(len(relays)):
        extra_array.append(relays[a][0])
    for i in range(meshes):
        for j in range(meshes):
            xi = (i % chosen_grid) * 20 + (20 / 2)
            yi = (i // chosen_grid) * 20 + (20 / 2)
            xj = (j % chosen_grid) * 20 + (20 / 2)
            yj = (j // chosen_grid) * 20 + (20 / 2)
            if i != j:
                if (xi, yi) in extra_array and (xj, yj) in extra_array:
                    if (xi, yi) in sentinels and (xj, yj) in sentinels:
                        pass
                    else:
                        if math.dist((xi, yi), (xj, yj)) < 30:
                            graph[i][j] = 1  # Assigning a weight of 1 for simplicity

    # Apply Bellman-Ford algorithm with heuristic
    dist = [math.inf] * meshes
    dist[start] = 0

    # Relax edges selectively based on heuristic
    for _ in range(meshes):
        for u in range(meshes):
            for v in range(meshes):
                if graph[u][v] != math.inf:  # If there's an edge between u and v
                    if dist[u] + graph[u][v] < dist[v] and heuristic((v % chosen_grid) * 20 + (20 / 2), (v // chosen_grid) * 20 + (20 / 2), sink[0], sink[1]) < heuristic((u % chosen_grid) * 20 + (20 / 2), (u // chosen_grid) * 20 + (20 / 2), sink[0], sink[1]):
                        dist[v] = dist[u] + graph[u][v]

    # Acquire distances of only sentinels and calculate the sum of them
    sentinel_bf_heuristic = []
    cal_bf_heuristic = 0
    for i in range(meshes):
        xi = (i % chosen_grid) * 20 + (20 / 2)
        yi = (i // chosen_grid) * 20 + (20 / 2)
        if (xi, yi) in sentinels:
            if dist[i] == math.inf:  # Check if a sentinel is unreachable
                sentinel_bf_heuristic.append(999)  # Set distance to 0 if unreachable
                cal_bf_heuristic += 999
            else:
                sentinel_bf_heuristic.append(dist[i])
                cal_bf_heuristic += dist[i]

    return dist, sentinel_bf_heuristic, cal_bf_heuristic

def heuristic(current_x, current_y, sink_x, sink_y):
    # Define a heuristic function that estimates the distance from current_vertex to sink
    return math.dist((current_x, current_y), (sink_x, sink_y))

def len_sinked_relays(sinked_relays):
    len_sinked_relay = []
    for relay in sinked_relays:
        len_sinked_relay.append(relay[0])
    unique_tuples = set(len_sinked_relay) 
    return len(unique_tuples)

def len_free_slots(grid, sinked_relays):
    chosen_grid = int(grid / 20)
    return ((chosen_grid - 2)*(chosen_grid - 2) - len_sinked_relays(sinked_relays) - 1)

def sentinel_relay(sentinel_bman):
    sentinel_relays = []
    for i in range(len(sentinel_bman)):
        relay = sentinel_bman[i] - 1
        sentinel_relays.append(relay)
    return sentinel_relays

def get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, mesh_size, alpha, beta):
    sentinel_relays = sentinel_relay(sentinel_bman)
    if not sentinel_relays:  # Check if sentinel distances list is empty (all unreachable)
        calculate_performance = 0
        calculate_hops = 0
    else:
        #calculate_performance = (0.5 * len(sinked_relays)) + (0.5 * cal_bman)
        calculate_performance = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size,  alpha, beta)
        calculate_hops = sum(sentinel_relays) / len(sentinel_relays)
    calculate_relays = len_sinked_relays(sinked_relays)
    return calculate_performance, calculate_relays, calculate_hops

def epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size, alpha, beta): 

    distance_bman, sentinel_bman, cal_bman = dijkstra(grid, sink, sinked_relays, sinked_sentinels)
    epsilon = int(grid / 20)
    
    print(f'\nSentinel dijkstra: {sentinel_bman}\n')
    performance = ((alpha * len_sinked_relays(sinked_relays)) + (beta * (get_Diameter(sentinel_bman, cal_bman, mesh_size))))

    if (999 in sentinel_bman) or (get_Diameter(sentinel_bman, cal_bman, mesh_size) > epsilon ):
        return performance + 999
    else:
        return performance

def bellman_ford_worker(queue, sentinel_indices, graph, chosen_grid, sentinels):
    INF = float('inf')
    n = len(graph)
    dist = [INF] * n

    for i in sentinel_indices:
        dist[i] = 0

    for _ in range(n - 1):
        for u in sentinel_indices:
            for v in range(n):
                if graph[u][v] != 0:
                    if dist[u] + graph[u][v] < dist[v]:
                        dist[v] = dist[u] + graph[u][v]

    sentinel_bman = []
    cal_bman = 0
    for i in sentinel_indices:
        xi = (i % chosen_grid) * 20 + (20 / 2)
        yi = (i // chosen_grid) * 20 + (20 / 2)
        if (xi, yi) in sentinels:
            if dist[i] == INF:
                sentinel_bman.append(999)
                cal_bman += 999
            else:
                sentinel_bman.append(dist[i])
                cal_bman += dist[i]

    queue.put((sentinel_bman, cal_bman))

def bellman_ford_parallel(grid, free_slots, sink, relays, sentinels):
    chosen_grid = int(grid / 20)
    meshes = chosen_grid * chosen_grid

    graph = [[0 for _ in range(meshes)] for _ in range(meshes)]
    extra_array = [sink]
    extra_array.extend(sentinels)
    for relay, _ in relays:
        extra_array.append(relay)
    for i in range(meshes):
        for j in range(meshes):
            xi = (i % chosen_grid) * 20 + (20 / 2)
            yi = (i // chosen_grid) * 20 + (20 / 2)
            xj = (j % chosen_grid) * 20 + (20 / 2)
            yj = (j // chosen_grid) * 20 + (20 / 2)
            if i != j:
                if (xi, yi) in extra_array and (xj, yj) in extra_array:
                    if (xi, yi) in sentinels and (xj, yj) in sentinels:
                        pass
                    else:
                        if math.dist((xi, yi), (xj, yj)) < 30:
                            graph[i][j] = 1

    num_processes = 4
    queue = Queue()
    processes = []

    sentinel_indices = [i for i in range(meshes) if (i % chosen_grid == 0 or i % chosen_grid == chosen_grid - 1 or i < chosen_grid or i >= meshes - chosen_grid)]

    chunk_size = len(sentinel_indices) // num_processes

    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = start_index + chunk_size if i < num_processes - 1 else len(sentinel_indices)
        process = Process(target=bellman_ford_worker, args=(queue, sentinel_indices[start_index:end_index], graph, chosen_grid, sentinels))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    sentinel_distances = []
    cal_bman = 0
    for _ in range(num_processes):
        sentinel_bman, partial_cal_bman = queue.get()
        sentinel_distances.extend(sentinel_bman)
        cal_bman += partial_cal_bman

    return sentinel_distances, cal_bman


def floyd_warshall_paths_matlab(grid, sink, sentinels, relays):

    chosen_grid = int(grid / 20)
    meshes = chosen_grid * chosen_grid

    # Creating the graph using the 
    graph = [[0 for _ in range(meshes)] for _ in range(meshes)]
    extra_array = [sink]
    extra_array.extend(sentinels)
    for relay, _ in relays:
        extra_array.append(relay)
    for i in range(meshes):
        for j in range(meshes):
            xi = (i % chosen_grid) * 20 + (20 / 2)
            yi = (i // chosen_grid) * 20 + (20 / 2)
            xj = (j % chosen_grid) * 20 + (20 / 2)
            yj = (j // chosen_grid) * 20 + (20 / 2)
            if i != j:
                if (xi, yi) in extra_array and (xj, yj) in extra_array:
                    if (xi, yi) in sentinels and (xj, yj) in sentinels:
                        pass
                    else:
                        if math.dist((xi, yi), (xj, yj)) < 30:
                            graph[i][j] = 1

    eng = matlab.engine.start_matlab()
    adj_max_matlab = matlab.double(graph)
    D, P = eng.FloydSPR2(adj_max_matlab, nargout=2)
    eng.quit()
    
    return np.array(D), np.array(P)


def bellman_ford_matlab(grid, sink, relays, sentinels):
    chosen_grid = int(grid / 20)
    meshes = chosen_grid * chosen_grid
    start = -1
    for i in range(meshes):
        x = (i % chosen_grid) * 20 + (20 / 2)
        y = (i // chosen_grid) * 20 + (20 / 2)
        if (x, y) == sink:
            start = i
            break

    # Create adjacency matrix
    graph = [[0 for _ in range(meshes)] for _ in range(meshes)]
    extra_array = [sink]
    extra_array.extend(sentinels)
    for relay, _ in relays:
        extra_array.append(relay)
    for i in range(meshes):
        for j in range(meshes):
            xi = (i % chosen_grid) * 20 + (20 / 2)
            yi = (i // chosen_grid) * 20 + (20 / 2)
            xj = (j % chosen_grid) * 20 + (20 / 2)
            yj = (j // chosen_grid) * 20 + (20 / 2)
            if i != j:
                if (xi, yi) in extra_array and (xj, yj) in extra_array:
                    if (xi, yi) in sentinels and (xj, yj) in sentinels:
                        pass
                    else:
                        if math.dist((xi, yi), (xj, yj)) < 30:
                            graph[i][j] = 1

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Add the directory containing the MATLAB script to the MATLAB path
    directory_path = 'C:/Users/nouri/OneDrive/Desktop/Papers/Python program files/Python program files/PersonalModules'
    eng.addpath(directory_path)
    
    # Convert Python lists to MATLAB arrays
    graph_matlab = matlab.double(graph)
    # Convert Python tuples to a cell array in MATLAB
    sentinels_cell = matlab.double(sentinels)
    # print(sentinels_cell)

    # Call MATLAB function
    sentinel_bman_cell = eng.bellman_ford_mat(chosen_grid, meshes, start, graph_matlab, sentinels_cell)
    
    # Convert the MATLAB result to a Python list
    sentinel_bman = [int(cell) if cell != 999 else 999 for cell in sentinel_bman_cell] 
    cal_bman = max(sentinel_bman)
    
    # Stop MATLAB engine
    eng.quit()    

    return sentinel_bman, cal_bman