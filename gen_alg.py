import random
from itertools import combinations
import numpy as np
import ast
from statistics import mean
from math import sqrt
np.set_printoptions(legacy="1.25")

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
    intersection = np.sum((v1 > 0) & (v2 > 0))
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


def psim_user(u, targetuser, ratings):
    rated_tu = list(ratings[ratings["userID"] == targetuser]["ISBN"])
    rated_u = list(ratings[ratings["userID"] == int(u)]["ISBN"])
    same_books = set(rated_tu).intersection(rated_u)
    if len(same_books) > 0:
        for book in same_books:
            rating_tu = ratings[
                (ratings["ISBN"] == book) & (ratings["userID"] == targetuser)
            ]["bookRating"].values[0]
            rating_u = ratings[
                (ratings["ISBN"] == book) & (ratings["userID"] == u)
            ]["bookRating"].values[0]
            mean_tu = mean(
                    list(ratings[ratings["userID"] == targetuser]["bookRating"])
            )
            mean_u = mean(list(ratings[ratings["userID"] == u]["bookRating"]))
            numerator = (rating_tu - mean_tu) * (rating_u - mean_u)
            denominator = sqrt((rating_tu - mean_tu) ** 2) * sqrt(
                (rating_u - mean_u) ** 2
            )
            if denominator != 0:
                psim = numerator / denominator
                return psim
            else:
                return 0
    else:
        return 0


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


def crossover(bestMemdf):
    newpop = []
    df_list = list(bestMemdf['Individual'])
    for _ in range(len(df_list)):
        pair = random.sample(df_list, 2)
        children = random.sample(pair[0], 3) + random.sample(pair[1], 3)
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


def predict(ratings, bestmem, targetuser):
    predict_score = []
    psim_sc = []
    for individual in bestmem:
        ind_score = []
        for i in individual:
            users = list(ratings[ratings['ISBN'] == i]['userID'])
            if len(users) > 0:
                num_sum = 0
                psim_scores = 0
                for u in users:
                    rating_u = ratings[
                        (ratings["ISBN"] == i) & (ratings["userID"] == u)
                    ]["bookRating"].values[0]
                    mean_u = mean(list(ratings[ratings["userID"] == u]["bookRating"]))
                    psim_u = psim_user(u, targetuser, ratings)
                    numerator = (rating_u - mean_u) * psim_u
                    psim_scores += psim_u
                    num_sum += numerator
                book_score = num_sum / psim_scores
                ind_score.append(book_score)
                psim_sc.append(psim_scores)
            else:
                ind_score.append(0)
        ind_total = sum(ind_score)
        predict_score.append(ind_total)
    
    return predict_score, psim_sc




# def crossover2():
# another way of doing the crossover
