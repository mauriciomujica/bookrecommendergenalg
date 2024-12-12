import random
from itertools import combinations
import numpy as np


def initialPop(targetUser, ratings, books, M, N):
    rated_items = ratings[ratings["userID"] == targetUser]["ISBN"].tolist()
    all_items = set(books["ISBN"].tolist())
    unrated_items = list(all_items - set(rated_items))

    population = []
    for _ in range(M):
        individual = random.sample(unrated_items, N)
        population.append(individual)

    return population


def jaccard_similarity(v1, v2):
    try:
        intersection = np.sum((v1 > 0) & (v2 > 0))
    except ValueError:
        print(v1)
        print(v2)
    union = np.sum((v1 > 0) | (v2 > 0))
    return intersection / (union + intersection) if union != 0 else 0


def correlationCal(pop, books):
    books2 = books.set_index("ISBN")
    fitness_scores = []

    for z in pop:
        vectors = []
        for item in z:
            if item in books2.index:
                vectors.append(books2.loc[item].to_numpy())
            else:
                raise ValueError(f"Item '{item}' not found in the dataframe.")
        correlations = []
        for i1, i2 in combinations(range(len(vectors)), 2):
            corr = jaccard_similarity(vectors[i1], vectors[i2])
            correlations.append(corr)

        fitness_value = sum(correlations)
        fitness_scores.append(fitness_value)

    return fitness_scores


def crossover(bestMem):
    for i in range(len(bestMem), 2):
        print(i)
