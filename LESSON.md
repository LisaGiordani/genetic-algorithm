Introduction
============

> Metaheuristics are mathematical optimization algorithms solving $\argmin_{x \in X} f(x)$.
>
> Synonyms:
>
> - search heuristics,
> - evolutionary algorithms,
> - stochastic local search.
>
> The general approach is to only look at the solutions, by trial and error, without further information on its structure.
> Hence the problem is often labelled as "black-box".
>
> Link to NP-hardness/curse of dimensionality: easy to evaluate, hard to solve.
> Easy to evaluate = fast, but not as fast as the algorithm itself.
> Hard to solve, but not impossible.


Hard optimization
-----------------

Metaheuristics are algorithms which aim at solving "hard" mathematical optimization problems.

A mathematical optimization problem is defined by a "solution" $x$
and an "objective function" $f:x\mapsto\reals$ :
$$\argmin_{x \in X} f(x)$$

One can consider using $\argmax$ without loss of genericity[^argminmax].
Usually, the set $X$ is defined intentionally and constraints on $x$ are managed separately.
For example, using a function $g:x\mapsto \{0,1\}$:
$$\argmin_{x\in[0,1]^n} f(x),\quad\text{s.t.}\quad g(x)=0$$



[^argminmax]: In the metaheuristics literature, $\argmax$ is often assumed for evolutionary algorithms, whether $\argmin$ is often assumed for local search or simulated annealing.



Complete VS approximation VS heuristics.


Algorithmics
============

> Those algorithms are randomized and iteratives (hence stochastics) and manipulates a sample (synonym population)
> of solutions (s. individual) to the problem, each one being associated with its quality (s. cost, fitness).
>
> Thus, algorithms have a main loop, and articulate functions that manipulates the sample (called "operators").
>
> Main design problem: exploitation/exploration compromise (s. intensification/diversification).
> Main design goal: raise the abstraction level.
> Main design tools: learning (s. memory) + heuristics (s. bias).
>
> Forget metaphors and use mathematical descriptions.
>
> Seek a compromise between complexity, performances and explainability.
>
> The is no better "method".
> Difference between model and instance, for problem and algorithm.
> No Free Lunch Theorem.
> But there is a "better algorithm instances on a given problem instances set".
>
> The better you understand it, the better the algorithm will be.


Naive algorithms
----------------

- Enumeration
- Grid search
- Random search
- Low-discrepency random number generators
- Random walk

- Stochastic convergence: there is a non-zero probability to sample the optimum
  after a finite number of steps.
- Conventional convergence: a stochastic search will tend to stabilize over time
  on a (better) objective function value.

- Ergodicity
- Quasi-ergodicity
- Necessary condition for stochasitc convergence: quasi-ergodicity.

- When is a random walk convergent?
- Example: 2D fixed-step size random walk.


Descent Algorithms
-----------------

Generic template:
```python
x = None
p = uniform(xmin,xmax)
for i in range(g):
    x = select(x,p)
    p = variation(x)
    p = evaluate(p)
```

Greedy algorithm:
```python
def select(x,p):
    if better(f(x),f(p)):
        return x
    else:
        return p
```

What are the conditions for which a greedy algorithm would converge?


Simulated Annealing
-------------------

```python
def select(x,p):
    if f(x) < f(p) or uniform(0,1) <= exp(-1/T*(f(p)-f(x))):
        return x
    else:
        return p
    T = decrease(T)
```

What occurs when T is high? When it is low?

Relationship to Metropolis-Hastings algorithm.
Sampling in a parametrized approximation of the objective function
(i.e. from uniform to Dirac(s)).


Evolutionary Algorithms
-----------------------

Generic template:
```python
P = uniform(xmin,xmax,n)
for i in range(g):
    parents = selection(P)
    offsprings = variation(parents)
    P = replacement(parents,offsprings)
```

More complete template:
```python
def evol(f):
    opt = float('inf')
    P = uniform(xmin,xmax,n)
    for i in range(g):
        parents = selection(P)
        offsprings = variation(parents)
        offsprings = evaluate(offsprings,f)
        opt = best(parents,offsprings)
        P = replacement(parents,offsprings)
    return best(P)
```

Evolution Strategies (ES), numerical space:
```python
def variation(parents):
    P = []
    for x in parents:
        P.append( x + normal(mean,variance) )
```

Genetic Algorithm (GA), boolean space:
```python
def variation(parents):
    crossed = crossover(parents)
    mutated = mutation(crossed)
    return mutated
```

Is ES convergent?
If $$#P=1$$, what are the differences with a random walk?


Estimation of Distribution Algorithms
-------------------------------------

```python
def variation(parents, law, n):
    parameters = estimate(parents, law)
    offsprings = sample(paramaters, law, n)
    return offsprings
```

What would be an example for numerical problems?
For boolean problems?
How to ensure convergence?


Ant Colony Algorithms
---------------------

TODO


Problem modelization
====================

> Way to assess the quality: fitness function.
> Way to model a solution: encoding.


Main models
-----------

> Encoding:
>
> - continuous (s. numeric),
> - discrete metric (integers),
> - combinatorial (graph, permutation).
>
> Fitness:
>
> - mono-objective,
> - multi-modal,
> - multi-objectives (cf. Pareto optimality).


Constraints management
----------------------

> Main constraints management tools for operators:
> - penalization,
> - reparation,
> - generation.


Performance evaluation
======================

What is performance
-------------------

> Main performances axis:
>
> - time,
> - quality,
> - probability.
>
> Additional performance axis:
>
> - robustness (cf. "external" problem structure),
> - stability (cf. "internal" randomisation).
>
> Golden rule: the output of a metaheuristic is a distribution, not a solution.


### Run time and target quality

One may think that the obvious objective of an optimization algorithm is to find the location of the optimum.
While this is true for deterministic and provable optimization, it is more complex in the case of metaheuristics.
When dealing with search heuristics, the quality of the (sub)optimum found is also a performance metric,
as one want to maximize the quality of the best solution found during the search.

The two main performance metrics are thus the runtime that is necessary to find a solution of a given quality,
and conversely, the quality of the solution which can be found in a given runtime.
Of course, those two metrics tend to be contradictory: the more time is given to the search, the best the solution,
and conversely, the better the solution one want, the longer the run should be.


### Measuring time and quality

To measure the run time, a robust measure of time should be available.
However, measuring run time on a modern computer is not necessarily robust, for instance because one cannot easily
control the context switching managed by the scheduler, or because the CPU load can produce memory access contentions.
However, in practical application, the call to the objective function largely dominates any other part of the algorithm.
The number of calls to the objective function is thus a robust proxy measure of the run time.

To measure the quality of solutions, the obvious choice is to rely on absolute values.
However, this may vary a lot across problem instances and be read differently if one has a minimization or a
maximization problem.
It may thus be useful to use the error against a known bound of the problem.


### Probability of attainment

For metaheuristics which are based on randomized process (which is the vast majority of them), measuring time and
quality is not enough to estimate their performance.
Indeed, if one run several time the same "algorithm", one will get different results, hence different performances.

That's why it's more useful to consider that the "output" of an "algorithm" is not a single solution, but a random
variable, a distribution of several solutions.
If one define an "algorithm" with fuzzy concepts, like "simulated annealing" or "genetic algorithm", the reason is
obvious, because the terms encompass a large variety of possible implementations.
But one should keep in mind that even a given implementation has (a lot of) parameters, and that metaheuristics are
usually (very) sensitive to their parameter setting.

In order to have a good mental image of how to asses the performance of a solver,
one should relize that we can only *estimates* the performances, considering *at least* run time, quality
*and probability* of attaining a fixed target.


### Robustness and stability


Empirical evaluation
--------------------

> Proof-reality gap is huge, thus empirical performance evaluation is gold standard.
>
> Empirical evaluation = scientific method.
>
> Basic rules of thumb:
>
> - randomized algorithms => repetition of runs,
> - sensitivity to parameters => design of experiments,
> - use statistical tools,
> - design experiments to answer a single question,
> - test one thing at a time.

## Useful statistical tools

> Statistical tests:
>
> - classical null hypothesis: test equality of distributions.
> - beware of p-value.
>
> How many runs?
>
> - not always "as many as possible",
> - maybe "as many as needed",
> - generally: 15 (min for non-parametric tests) -- 20 (min for parametric-gaussian tests).
>
> Use robust estimators: median instead of mean, Inter Quartile Range instead of standard deviation.


## Expected Empirical Cumulative Distribution Functions

> On Run Time: ERT-ECDF.
>
> $$ERTECDF(\{X_0,\dots,X_i,\dots,X_r\}, \delta, f, t) := \#\{x_t \in X_t | f(x_t^*)>=\delta \}$$
>
> $$\delta \in \left[0, \max_{x \in \mathcal{X}}(f(x))\right]$$
>
> $$X_i := \left\{\left\{ x_0^0, \dots, x_i^j, \dots, x_p^u | p\in[1,\infty[ \right\} | u \in [0,\infty[ \right\} \in \mathcal{X}$$
>
> with $p$ the sample size, $r$ the number of runs, $u$ the number of iterations, $t$ the number of calls to the objective
> function.
>
> The number of calls to the objective function is a good estimator of time because it dominates all other times.
>
> The dual of the ERT-ECDF can be easily computed for quality (EQT-ECDF).
>
> 3D ERT/EQT-ECDF may be useful for terminal comparison.


## Other tools

> Convergence curves: do not forget the golden rule and show distributions:
>
> - quantile boxes,
> - violin plots,
> - histograms.


Algorithm Design
================

## Neighborhood

> Convergence definition(s):
>
> - conventional,
> - stochastic convergence.
>
> Neighborhood: subset of solutions atteinable after an atomic transformation:
>
> - ergodicity,
> - quasi-ergodicity.
>
> Relationship to metric space in the continuous domain.


## Structure of problem/algorithms

> Structure of problems to exploit:
>
> - locality (basin of attraction),
> - separability,
> - gradient,
> - funnels.
>
> Structure with which to capture those structures:
>
> - implicit,
> - explicit,
> - direct.
>
> Silver rule: choose the algorithmic template that adhere the most to the problem model.
>
> - taking constraints into account,
> - iterate between problem/algorithm models.


## Grammar of algorithms

> Parameter setting < tuning < control.
>
> Portfolio approaches.
> Example: numeric low dimensions => Nelder-Mead Search is sufficient.
>
> Algorithm selection.
>
> Algorithms are templates in which operators are interchangeable.
>
> Most generic way of thinking about algorithms: grammar-based algorithm selection with parameters.
> Example: modular CMA-ES.

Parameter setting tools:

- ParamILS,
- SPO,
- i-race.

Design tools:

- ParadisEO,
- IOH profiler
- jMetal,
- Jenetics,
- ECJ,
- DEAP,
- HeuristicLab.


## Landscape-aware algorithms

> Fitness landscapes: structure of problems as seen by an algorithm.
> Features: tool that measure one aspect of a fitness landscape.
>
> We can observe landscapes, and learn which algorithm instance solves it better.
> Examples: SAT, TSP, BB.
>
> Toward automated solver design.

