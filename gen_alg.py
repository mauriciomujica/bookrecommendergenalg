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
        # Combine and shuffle books from both parents
        combined_books = list(set(pair[0] + pair[1]))
        # Ensure we have enough unique books to sample
        if len(combined_books) >= 6:
            children = random.sample(combined_books, 6)
            if children not in newpop:
                newpop.append(children)
        # If not enough unique books, skip this crossover
    return newpop


def similarityCal(ratings, newpop, sim_users):
    sim_scores = []
    for individual in newpop:
        sim_value = 0
        for book in individual:
            if book not in ratings["ISBN"].values:
                sim_value += 0
            else:
                users = list(ratings.index[ratings["ISBN"] == book])
                filtered_users = [user for user in users if user in sim_users.index]
                for user in filtered_users:
                    value = sim_users.loc[user].values[0]
                    sim_value += value
        sim_scores.append(sim_value)
    return sim_scores


def predict(ratings, sim_users, final_mem, targetUser):
    predict_score = []
    df_list = list(final_mem["Individual"])

    mean_tu = mean(ratings.loc[targetUser]["bookRating"])

    for individual in df_list:
        predict_value = 0
        for i in individual:
            users = list(ratings.index[ratings["ISBN"] == i])
            filtered_users = [user for user in users if user in sim_users.index]
            if not filtered_users:
                continue
            for user in filtered_users:
                try:
                    rating_u = ratings.loc[user][ratings.loc[user]["ISBN"] == i][
                        "bookRating"
                    ].values[0]
                except IndexError:
                    continue
                mean_u = mean(ratings.loc[user]["bookRating"])
                sim_value = sim_users.loc[user].values[0]
                if sim_value == 0:
                    continue
                numerator = (rating_u - mean_u) * sim_value
                result = mean_tu + numerator / sim_value
                predict_value += result
        predict_score.append(predict_value)

    return predict_score
