from docplex.mp.model import Model

def optimize_network(num_positions, max_hops_number):
    # Create a model
    model = Model(name="sensor_network_optimization")

    # Decision varables
    x = model.binary_var_dict(range(1, num_positions + 1), name="x")  # Node sensor deployed at position i
    r = model.binary_var_dict(range(1, num_positions + 1), name="r")  # Node relay deployed at position i
    y = model.binary_var_dict(range(1, max_hops_number + 1), name="y")  # Node relay deployed at k hops from Sink
    z = model.binary_var_dict(range(1, num_positions + 1), name="z")  # Node sensor deployed at position i
    t = model.binary_var_dict(range(1, num_positions + 1), name="t")  # Node relay deployed at position i

    # Objective function
    model.minimize(model.sum(y[k] for k in range(1, max_hops_number)))

    # Constraints
    # Constraint (1)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(x[i] + r[i] <= 1, ctname='constraint1_' + str(k) + '_' + str(i))

    # Constraint (2)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(z[i] >= x[i], ctname='constraint2_' + str(k) + '_' + str(i))

    # Constraint (3)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(y[k] >= model.sum(r[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == 1),
                                 ctname='constraint3_' + str(k) + '_' + str(i))

    # Constraint (4)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(r[i] >= model.sum(r[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == 1),
                                 ctname='constraint4_' + str(k) + '_' + str(i))

    # Constraint (5)
    model.add_constraint(x[1] + r[1] == 1, ctname='constraint5')

    # Constraint (6)
    for k in range(1, max_hops_number + 1):
        model.add_constraint(model.sum(r[i] * (k - 1) for i in range(1, num_positions + 1)) == model.sum(y[j] * j for j in range(1, k + 1)),
                             ctname='constraint6_' + str(k))

    # Constraint (7)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(y[k] <= model.sum(r[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == 1),
                                 ctname='constraint7_' + str(k) + '_' + str(i))

    # Constraint (8)
    for k in range(2, max_hops_number + 1):
        model.add_constraint(y[k - 1] >= y[k], ctname='constraint8_' + str(k))

    # Constraint (9)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(z[i] <= x[i], ctname='constraint9_' + str(k) + '_' + str(i))

    # Constraint (10)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(x[i] <= model.sum(z[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == k),
                                 ctname='constraint10_' + str(k) + '_' + str(i))

    # Constraint (11)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(r[i] <= model.sum(t[j] for j in range(1, num_positions + 1) if j != i and abs(j - i) == k),
                                 ctname='constraint11_' + str(k) + '_' + str(i))

    # Constraint (12)
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            model.add_constraint(t[i] <= r[i], ctname='constraint12_' + str(k) + '_' + str(i))

    # Constraint (13)
    model.add_constraint(model.sum(z[i] for i in range(1, num_positions + 1)) <= available_sensor_nodes, ctname='constraint13')

    # Solve the model
    model.solve()

    # Prnt solution
    print("Solution:")
    for k in range(1, max_hops_number + 1):
        for i in range(1, num_positions + 1):
            print("y[", k, "] = ", y[k].solution_value)
    print("Objective value: ", model.objective_value)

# Set the values for num_positions and max_hops_number
num_positions = 10
max_hops_number = 5
available_sensor_nodes = 15  # Replace with the actual number of available sensor nodes

# Call the function to solve the optimization problem
optimize_network(num_positions, max_hops_number)