import random
from itertools import combinations
import numpy as np
import ast
from statistics import mean
from math import sqrt


def initialPop(targetUser, ratings, books, M, N):
    rated_items = ratings[ratings["userID"] == targetUser]["ISBN"].tolist()
    all_items = set(books["ISBN"].tolist())
    unrated_items = list(all_items - set(rated_items))

    population = []
    for _ in range(M):
        individual = random.sample(unrated_items, N)
        population.append(individual)

    return population


def jaccardBooks(v1, v2):
    try:
        intersection = np.sum((v1 > 0) & (v2 > 0))
    except ValueError:
        print(v1)
        print(v2)
    union = np.sum((v1 > 0) | (v2 > 0))
    return intersection / (union + intersection) if union != 0 else 0


def psim(targetuser, flat_users, ratings):
    rated_targetuser = list(ratings[ratings["userID"] == targetuser]["ISBN"])
    psim_value = 0

    for user in flat_users:
        same_books_rated = []
        rated_user = list(ratings[ratings["userID"] == int(user)]["ISBN"])

        for book in rated_user:
            if book in rated_targetuser:
                same_books_rated.append(book)

        if len(same_books_rated) > 0:
            for i in same_books_rated:
                rating_tu = ratings[
                    (ratings["ISBN"] == i) & (ratings["userID"] == targetuser)
                ]["bookRating"].values[0]
                rating_u = ratings[
                    (ratings["ISBN"] == i) & (ratings["userID"] == user)
                ]["bookRating"].values[0]
                mean_tu = mean(
                    list(ratings[ratings["userID"] == targetuser]["bookRating"])
                )
                mean_u = mean(list(ratings[ratings["userID"] == user]["bookRating"]))

                numerator = (rating_tu - mean_tu) * (rating_u - mean_u)
                denominator = sqrt((rating_tu - mean_tu) ** 2) * sqrt(
                    (rating_u - mean_u) ** 2
                )
                if denominator != 0:
                    value = numerator / denominator
                    psim_value += value
                else:
                    psim_value += 0
        else:
            psim_value += 0
    return psim_value


def jaccardUsers(targetuser, flat_users, ratings):
    rated_targetuser = list(ratings[ratings["userID"] == targetuser]["ISBN"])
    jaccard_value = 0

    for user in flat_users:
        same_books_rated = []
        rated_user = list(ratings[ratings["userID"] == int(user)]["ISBN"])

        for book in rated_user:
            if book in rated_targetuser:
                same_books_rated.append(book)

        if len(same_books_rated) > 0:
            numerator = abs(len(same_books_rated))
            denominator = abs(len(rated_targetuser) + len(rated_user))
            value = numerator / denominator
            jaccard_value += value

    return jaccard_value


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
            corr = jaccardBooks(vectors[i1], vectors[i2])
            correlations.append(corr)

        fitness_value = sum(correlations)
        fitness_scores.append(fitness_value)

    return fitness_scores


def crossover(bestMem):
    newpop = []
    for _ in range(len(bestMem)):
        pair = random.sample(bestMem, 2)
        i1, i2 = [i for i in pair]
        l1 = ast.literal_eval(i1)
        l2 = ast.literal_eval(i2)
        children = random.sample(l1, 3) + random.sample(l2, 3)
        newpop.append(children)

    return newpop


def similarityCal(ratings, newpop, user):
    sim_scores = []
    for individual in newpop:
        total_users = []
        for i in individual:
            users = list(ratings[ratings["ISBN"] == i]["userID"])
            if len(users) > 0:
                total_users.append(users)
        try:
            flat_users = list(np.concatenate(total_users))
        except ValueError:
            print("individual that failed: ", individual)
        psim_ind = psim(user, flat_users, ratings)
        jac_ind = jaccardUsers(user, flat_users, ratings)
        sim_of_ind = psim_ind * jac_ind
        sim_scores.append(sim_of_ind)

    return sim_scores


# def crossover2():
# another way of doing the crossover
