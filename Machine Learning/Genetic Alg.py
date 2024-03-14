import random
from random import randint
import numpy as np
from mlp_feedforward import MLPFeedfoward
from sklearn.metrics import accuracy_score


# Fitness function
def evaluate_solution(population_size, X,y):
    for indiv in population_size:
        mlp = MLPFeedfoward()
        mlp.W[1] = np.reshape(indiv[0:16], ((4, 4)))
        mlp.W[2] = np.reshape(indiv[16:20], ((4, 1)))
        mlp.B[1] = np.reshape(indiv[20:24], ((1, 4)))
        mlp.B[2] = np.reshape(indiv[-1], ((1, 1)))

        y_pred = mlp.predict(X).round()
        acc = accuracy_score(y, y_pred)

# Generate an initial population
def initial_population(population_size, genome_length):
    '''
    Info:
        Function to create a population containing the number of individuals (population_size) passed as parameter.
    ----------
    Input:
        population_size: Number of individuals in the population (type: int)
    ----------
    Output:
        initial solution
    '''
    population = [[random.randint(1, 5) for _ in range(genome_length)] for _ in range(population_size)]
    return population


# Perform selection using tournament selection
def tournament_selection(population, X, y):
    '''
    Info:
    
    Function that selects the parents using the tournament method. 
    The tournament compares only two parents each time (k = 2).
    '''

    selected_parents = []
    
    for _ in range(len(population)):
        #tournament = random.sample(list(enumerate(population)), tournament_size)
        # Randomly selecting 2 individuals as parents
        indx_p1 = randint(len(population))
        indx_p2 = randint(len(population))


        # Calculating the fitness for each parent
        # Parent 1
        mlp_1 = MLPFeedfoward()
        
        # Reshaping the parent to match the weights shape
        mlp_1.W[1] = np.reshape(population[indx_p1][0:16], ((4, 4)))
        mlp_1.W[2] = np.reshape(population[indx_p1][16:20], ((4, 1)))
        mlp_1.B[1] = np.reshape(population[indx_p1][20:24], ((1, 4)))
        mlp_1.B[2] = np.reshape(population[indx_p1][-1], ((1, 1)))
        
        # Calculating the fitness (accuracy)
        y_pred_1 = mlp_1.predict(X).round()
        acc_1 = accuracy_score(y, y_pred_1)

        # Parent 2
        mlp_2 = MLPFeedfoward()
        # Reshaping the parent to match the weights shape
        mlp_2.W[1] = np.reshape(population[indx_p2][0:16], ((4, 4)))
        mlp_2.W[2] = np.reshape(population[indx_p2][16:20], ((4, 1)))
        mlp_2.B[1] = np.reshape(population[indx_p2][20:24], ((1, 4)))
        mlp_2.B[2] = np.reshape(population[indx_p2][-1], ((1, 1)))

        # Calculating the fitness (accuracy)
        y_pred_2 = mlp_2.predict(X).round()
        acc_2 = accuracy_score(y, y_pred_2)

        # Selecting the parent with the higher fitness (accuracy)
        if acc_1 > acc_2:
            selected_parents.append(population[indx_p1])
        else:
            selected_parents.append(population[indx_p2])
    return selected_parents

# Perform crossover
def crossover(parent1, parent2):
    '''
    Info:
        Funciton that realizes the crossover operation on the children
    
    Input:
        Parent: parent 1 and parent 2
    
    Outuput:
        Children: child 1 and child 2
    '''
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Perform mutation
def mutate(genome, mutation_rate):
    '''
    Info: 
        Funciton that realizes the mutation operation on the child
    
    Input:
        genome: Child genome
        mutation_rate: Mutation treshold value used
    
    Output:
        genome: genome of the child
    '''

    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] = random.randint(1, 5)
    return genome

# Genetic algorithm
def genetic_algorithm(population_size, genome_length, generations, mutation_rate, X, y):
    # Generate initial population
    population = initial_population(population_size, genome_length)
    
    for generation in range(generations):
        # Evaluate fitness of each genome
        fitness_values = [evaluate_solution(genome) for genome in population]
        
        # Perform selection
        selected_parents = tournament_selection(population, X, y)
        
        # Create offspring through crossover
        offspring = []
        for i in range(0, len(selected_parents), 2):
            child1, child2 = crossover(selected_parents[i], selected_parents[i + 1])
            offspring.extend([child1, child2])
        
        # Apply mutation
        mutated_offspring = [mutate(genome, mutation_rate) for genome in offspring]
        
        # Replace the population with the combined set of parents and offspring
        population = selected_parents + mutated_offspring
        
        # Output the best genome in the current generation
        best_genome = max(population, key=evaluate_solution)
        print(f"Generation {generation + 1}: Best Genome: {best_genome}, Fitness: {evaluate_solution(best_genome)}")
    
    # Output the best genome found after all generations
    best_genome = max(population, key=evaluate_solution)
    print(f"\nBest Genome Found: {best_genome}, Fitness: {evaluate_solution(best_genome)}")