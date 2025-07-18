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


def get_vectors(column, books):
    vectors = books.loc[column].to_numpy()
    return vectors


def correlationCal(book_vectors, N):

    correlations = []
    for i, j in combinations(range(N), 2):
        A = book_vectors[i]
        B = book_vectors[j]

        intersection = np.sum(A & B, axis=1)
        union = np.sum(A | B, axis=1)

        corr = np.divide(
            intersection,
            intersection + union,
            out=np.zeros_like(intersection, dtype=float),
            where=(intersection + union) != 0
        )
        correlations.append(corr)

    total_correlation = np.sum(correlations, axis=0)
    return total_correlation


def crossover(bestMemdf, R, N):
    newpop = []
    df_list = list(bestMemdf["Individual"])
    for _ in range(R):
        pair = random.sample(df_list, 2)
        combined_books = list(set(pair[0] + pair[1]))
        if len(combined_books) >= N:
            children = random.sample(combined_books, N)
            if children not in newpop:
                newpop.append(children)
    return newpop


def similarityCal(ratings, newpop, sim_users, c):
    sim_scores = []
    for individual in newpop:
        sim_value = 0
        for book in individual:
            if book in c:
                users = ratings.xs(book, level = 'ISBN').index
                #if users.size != 0:
                for user in users:
                    value = sim_users.loc[user].values[0]
                    sim_value += value
            else:
                sim_value += 0
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
