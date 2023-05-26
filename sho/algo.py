########################################################################
# Algorithms
########################################################################
import numpy as np

def random(func, init, again):
    """Iterative random search template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 0
    while again(i, best_val, best_sol):
        sol = init()
        val = func(sol)
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def greedy(func, init, neighb, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 1
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def recuit_simule(func, init, neighb, again, Tinit, beta):
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 1
    T = Tinit
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        alpha = np.random.rand()
        if val >= best_val: # cas où on améliore la couverture
            best_val = val
            best_sol = sol
        elif np.exp(-(best_val-val)/T) > alpha: # cas où on n'améliore pas la couverture mais on explore quand même
            best_val = val
            best_sol = sol
        i += 1
        T /= beta
    return best_val, best_sol

# TODO add a population-based stochastic heuristic template.
def evolution_strategies(func, init, neighb, again, nb_pop):
    population_sol = [init() for i in range(nb_pop)] # coordonnées des antennes de tous les individus de la pop
    population_val = [func(individual) for individual in population_sol]

    best_sol = population_sol[np.argmax(population_val)]
    best_val = max(population_val)

    i = nb_pop # nb d'appel à la fonction func à l'initialisation

    while again(i, best_val, best_sol):

        # selection : garde la meilleure moitié de la population
        sorted_population_index = sorted(range(nb_pop), key=lambda k: population_val[k])
        parents_sol = [population_sol[i] for i in sorted_population_index[round(nb_pop/2):]]
        parents_val = [population_val[i] for i in sorted_population_index[round(nb_pop/2):]]

        # variation
        children_sol = [neighb(parent) for parent in parents_sol]
        children_val = [func(child) for child in children_sol]

        # replace
        population_sol = children_sol + parents_sol
        population_val = children_val + parents_val

        # best solution
        best_sol = population_sol[np.argmax(population_val)]
        best_val = max(population_val)

        i += int(np.floor(nb_pop/2)) # nb d'appel à la fonction func pendant une itération
    return best_val, best_sol
