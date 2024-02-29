import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


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
        ax.plot(sentinels[i][0], sentinels[i][1], marker='s', color='orange')
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


# --------------------------------------------------------------------------------------------------------------


'''def bellman_ford(grid, free_slots, sink, relays, sentinels):
    chosen_grid = int(grid / 20)
    meshes = chosen_grid * chosen_grid
    start= -1
    for i in range(meshes):
        x = (i % chosen_grid) * 20 + (20 / 2)
        y = (i // chosen_grid) * 20 + (20 / 2)
        if (x, y) == sink:
            start = i
            break

    # Create adjacency matrix
    graph = [[0 for j in range(meshes)] for i in range(meshes)]
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

    INF = float('inf')

    # Initialize distance array with infinity for all vertices except the start vertex
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0

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
    for i in range(meshes):
        xi = (i % chosen_grid) * 20 + (20 / 2)
        yi = (i // chosen_grid) * 20 + (20 / 2)
        if (xi, yi) in sentinels:
            sentinel_bman.append(dist[i])
            cal_bman = cal_bman + dist[i]

    return dist, sentinel_bman, cal_bman


def get_stat(sinked_relays, sentinel_bman, cal_bman):
    calculate_performance = (0.5 * len(sinked_relays)) + (0.5 * cal_bman)
    calculate_relays = len(sinked_relays)
    calculate_hops = cal_bman / len(sentinel_bman)
    return calculate_performance, calculate_relays, calculate_hops'''

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

    # Create adjacency matrix
    graph = [[0 for j in range(meshes)] for i in range(meshes)]
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

    INF = float('inf')

    # Initialize distance array with infinity for all vertices except the start vertex
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0

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
    for i in range(meshes):
        xi = (i % chosen_grid) * 20 + (20 / 2)
        yi = (i // chosen_grid) * 20 + (20 / 2)
        if (xi, yi) in sentinels:
            if dist[i] == INF:  # Check if a sentinel is unreachable
                sentinel_bman.append(999)  # Set distance to 0 if unreachable
            else:
                sentinel_bman.append(dist[i])
                cal_bman += dist[i]

    return dist, sentinel_bman, cal_bman


def get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, mesh_size, alpha, beta):
    if not sentinel_bman:  # Check if sentinel distances list is empty (all unreachable)
        calculate_performance = 0
        calculate_hops = 0
    else:
        #calculate_performance = (0.5 * len(sinked_relays)) + (0.5 * cal_bman)
        calculate_performance = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size,  alpha, beta)
        calculate_hops = cal_bman / len(sentinel_bman)
    calculate_relays = len(sinked_relays)
    return calculate_performance, calculate_relays, calculate_hops

def epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, cal_bman, mesh_size, alpha, beta):
    epsilon = cal_bman

    # The cal_bman is now considered as the epsilon bound
    performance = ((alpha * len(sinked_relays)) + (beta * (cal_bman / mesh_size)))

    return performance
    '''# If the performance exceeds the epsilon bound (cal_bman), return 0
    if performance >= epsilon:
        return float('inf')
    else:
        # Otherwise, return the calculated performance
        return performance'''

'''
    Epsilon constraint method to return a pareto front
'''
'''def epsilon_constraint_method(grid, free_slots, sink, sinked_relays, sinked_sentinels, alpha, beta):
    pareto_front = []
    for epsilon in range(1, 11):  # Adjust the range of epsilon as needed
        distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)

        # The cal_bman is now considered as the epsilon bound
        performance = (alpha * len(sinked_relays)) + (beta * cal_bman)

        # If the performance does not exceed the epsilon bound (cal_bman), add to the Pareto front
        if performance <= epsilon:
            pareto_front.append((performance, cal_bman, len(sinked_relays)))

    return pareto_front'''
