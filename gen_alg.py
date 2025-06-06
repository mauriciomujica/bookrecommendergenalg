import random
from itertools import combinations
import numpy as np
from statistics import mean
from math import sqrt

np.set_printoptions(legacy="1.25")


def sim_matrix(targetUser, rated_items, ratings, similar_users, sim_df):
    for user in similar_users:
        psim = 0

        rated_u = list(ratings.loc[user]["ISBN"])
        same_books = list(set(rated_items).intersection(rated_u))
        for book in same_books:
            rating_u = ratings.loc[user][ratings.loc[user]["ISBN"] == book][
                "bookRating"
            ].values[0]
            rating_tu = ratings.loc[targetUser][
                ratings.loc[targetUser]["ISBN"] == book
            ]["bookRating"].values[0]
            mean_u = mean(ratings.loc[user]["bookRating"])
            mean_tu = mean(ratings.loc[targetUser]["bookRating"])

            numerator = (rating_tu - mean_tu) * (rating_u - mean_u)
            denominator = sqrt((rating_tu - mean_tu) ** 2) * sqrt(
                (rating_u - mean_u) ** 2
            )

            if denominator == 0:
                value = 0
            else:
                value = numerator / denominator

            psim += value

        jaccard_num = len(same_books)
        jaccard_den = len(rated_u) + len(rated_items)

        jaccard = jaccard_num / jaccard_den

        simU = psim * jaccard

        sim_df.at[user, targetUser] = simU

    return sim_df


def initialPop(rated_items, books, M, N):
    all_items = books.index.tolist()
    unrated_items = list(set(all_items) - set(rated_items))
    # sorted_items = sorted(unrated_items)

    population = []
    for _ in range(M):
        individual = random.sample(unrated_items, N)
        population.append(individual)

    return population


def jaccardBooks(v1, v2):
    intersection = np.sum((v1 > 0) & (v2 > 0))
    union = np.sum((v1 > 0) | (v2 > 0))
    return intersection / (union + intersection) if union != 0 else 0


def psim_user(u, targetuser, ratings):
    rated_tu = list(ratings[ratings["userID"] == targetuser]["ISBN"])
    rated_u = list(ratings[ratings["userID"] == u]["ISBN"])
    same_books = set(rated_tu).intersection(rated_u)
    if len(same_books) > 0:
        for book in same_books:
            rating_tu = ratings[
                (ratings["ISBN"] == book) & (ratings["userID"] == targetuser)
            ]["bookRating"].values[0]
            rating_u = ratings[(ratings["ISBN"] == book) & (ratings["userID"] == u)][
                "bookRating"
            ].values[0]
            mean_tu = mean(list(ratings[ratings["userID"] == targetuser]["bookRating"]))
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


def correlationCal(pop, books):
    fitness_scores = []

    for z in pop:
        vectors = []
        for item in z:
            vectors.append(books.loc[item].to_numpy())

        correlations = []
        for i1, i2 in combinations(range(len(vectors)), 2):
            corr = jaccardBooks(vectors[i1], vectors[i2])
            correlations.append(corr)

        fitness_value = sum(correlations)
        fitness_scores.append(fitness_value)

    return fitness_scores


def crossover(bestMemdf, R):
    newpop = []
    df_list = list(bestMemdf["Individual"])
    for _ in range(R):
        pair = random.sample(df_list, 2)
        children = random.sample(pair[0], 3) + random.sample(pair[1], 3)
        if children not in newpop:
            newpop.append(children)

    return newpop


def similarityCal(ratings, newpop, sim_users):
    sim_scores = []
    for individual in newpop:
        total_users = []
        for i in individual:
            users = list(ratings.index[ratings["ISBN"] == i])
            if len(users) > 0:
                total_users.append(users)
        try:
            flat_users = list(np.concatenate(total_users))
        except ValueError:
            newpop.remove(individual)

        filtered_users = [user for user in flat_users if user in sim_users.index]
        sim_value = 0

        if len(filtered_users) == 0:
            value = 0
            sim_value += value
        else:
            for user in filtered_users:
                value = sim_users.loc[user].values[0]
                sim_value += value

        sim_scores.append(sim_value)

    return sim_scores


def predict(ratings, bestmem, targetuser):
    predict_score = []
    for individual in bestmem:
        ind_score = []
        for i in individual:
            users = list(ratings[ratings["ISBN"] == i]["userID"])
            if len(users) > 0:
                num_sum = 0
                psim_scores = []
                for u in users:
                    rating_u = ratings[
                        (ratings["ISBN"] == i) & (ratings["userID"] == u)
                    ]["bookRating"].values[0]
                    mean_u = mean(list(ratings[ratings["userID"] == u]["bookRating"]))
                    psim_u = psim_user(u, targetuser, ratings)
                    numerator = (rating_u - mean_u) * psim_u
                    psim_scores.append(psim_u)
                    num_sum += numerator
                if sum(psim_scores) > 0:
                    book_score = num_sum / sum(psim_scores)
                else:
                    book_score = 0
                ind_score.append(book_score)
            else:
                ind_score.append(0)
        ind_total = sum(ind_score)
        predict_score.append(ind_total)

    return predict_score


# def crossover2():
# another way of doing the crossover
