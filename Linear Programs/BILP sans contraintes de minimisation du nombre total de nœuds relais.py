from docplex.mp.model import Model

def optimize_network(num_positions, max_hops_number):
    # Create a model
    model = Model(name="sensor_network_optimization")

    # Decision variables
    xi = model.binary_var_dict(range(1, num_positions + 1), name="xi")  # Node sensor deployed at position i
    ri = model.binary_var_dict(range(1, num_positions + 1), name="ri")  # Node relay deployed at position i
    yk = model.binary_var_dict(range(1, max_hops_number + 1), name="yk")  # Node relay deployed at k hops from Sink
    zi = model.binary_var_dict(range(1, num_positions + 1), name="zi")  # Node sensor deployed at position i
    ti = model.binary_var_dict(range(1, num_positions + 1), name="ti")  # Node relay deployed at position i

    # Objective function
    model.minimize(model.sum(yk[k] for k in range(1, max_hops_number + 1)))

    # Constraints
    # Constraint (1)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(xi[i] + ri[i] <= 1, ctname='constraint1_' + str(k) + '_' + str(i))

    # Constraint (2)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(zi[i] >= xi[i], ctname='constraint2_' + str(k) + '_' + str(i))

    # Constraint (3)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(yk[k] >= model.sum(ri[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == 1),
                                 ctname='constraint3_' + str(k) + '_' + str(i))

    # Constraint (4)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(ri[i] >= model.sum(ri[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == 1),
                                 ctname='constraint4_' + str(k) + '_' + str(i))

    # Constraint (5)
    model.add_constraint(xi[1] + ri[1] == 1, ctname='constraint5')

    # Constraint (6)
    for k in range(1, max_hops_number + 1):
        model.add_constraint(model.sum(ri[i] * (k - 1) for i in range(1, num_positions + 1)) == model.sum(yk[j] * j for j in range(1, k + 1)),
                             ctname='constraint6_' + str(k))

    # Constraint (7)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(yk[k] <= model.sum(ri[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == 1),
                                 ctname='constraint7_' + str(k) + '_' + str(i))

    # Constraint (8)
    for k in range(2, max_hops_number + 1):
        model.add_constraint(yk[k - 1] >= yk[k], ctname='constraint8_' + str(k))

    # Constraint (9)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(zi[i] <= xi[i], ctname='constraint9_' + str(k) + '_' + str(i))

    # Constraint (10)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(xi[i] <= model.sum(zi[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == k),
                                 ctname='constraint10_' + str(k) + '_' + str(i))

    # Constraint (11)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(ri[i] <= model.sum(ti[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == k),
                                 ctname='constraint11_' + str(k) + '_' + str(i))

    # Constraint (12)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(ti[i] <= ri[i], ctname='constraint12_' + str(k) + '_' + str(i))

    # Constraint (13)
    model.add_constraint(model.sum(zi[i] for i in range(1, num_positions + 1)) <= available_sensor_nodes, ctname='constraint13')

    # Solve the model
    model.solve()

    # Print solution
    print("Solution:")
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            print("yk[", k, "] = ", yk[k].solution_value)
    print("Objective value: ", model.objective_value)

# Set the values for num_positions and max_hops_number
num_positions = 10
max_hops_number = 5
available_sensor_nodes = 15  # Replace with the actual number of available sensor nodes

# Call the function to solve the optimization problem
optimize_network(num_positions, max_hops_number)
