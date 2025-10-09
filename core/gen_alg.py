import random
import pandas as pd
from itertools import combinations
import numpy as np
from statistics import mean
from math import sqrt


def sim_matrix(targetUser, rated_items, ratings, similar_users, sim_df):
    for user in similar_users:
        psim = 0
        numerator_sum = 0
        denominator_tu = 0
        denominator_u = 0

        rated_u = ratings.loc[user]["ISBN"]
        if isinstance(rated_u, (str, int)):
            rated_u = [rated_u]
        else:
            rated_u = list(rated_u)

        user_ratings = ratings.loc[(user)]
        if isinstance(user_ratings, pd.Series):
            user_ratings = user_ratings.to_frame().T
        mean_u = mean(user_ratings["bookRating"])
        mean_tu = mean(ratings.loc[targetUser]["bookRating"])
        
        same_books = list(set(rated_items).intersection(rated_u))
        for book in same_books:
            rating_u = user_ratings[user_ratings["ISBN"] == book]["bookRating"].values[0]
            rating_tu = ratings.loc[targetUser][
                ratings.loc[targetUser]["ISBN"] == book
            ]["bookRating"].values[0]

            tu_value = rating_tu - mean_tu
            u_value = rating_u - mean_u
            
            numerator = tu_value * u_value
            numerator_sum += numerator

            denominator_tu += tu_value
            denominator_u += u_value
            
        denominator = sqrt((denominator_tu) ** 2) * sqrt(
            (denominator_u) ** 2
        )

        if denominator == 0:
            value = 0
        else:
            value = numerator_sum / denominator

            psim += value

        jaccard_num = len(same_books)
        jaccard_den = len(rated_u) + len(rated_items)

        jaccard = jaccard_num / jaccard_den

        simU = psim * jaccard

        sim_df.at[user, targetUser] = simU

    return sim_df


def mean_users(similar_users, ratings_filtered):
    mean_list = []
    for user in similar_users:
        mean_u = mean(
            ratings_filtered[ratings_filtered["userID"] == user]["bookRating"]
        )
        mean_list.append(mean_u)
    return mean_list


def initialPop(rated_items, books, M, N):
    all_items = books.index.tolist()
    unrated_items = list(set(all_items) - set(rated_items))

    population = []
    for _ in range(M):
        individual = random.sample(unrated_items, N)
        population.append(individual)

    return population


def get_vectors(df, books):
    book_vectors = []
    for col in df.columns:
        try: 
            vectors = books.loc[df[col]].to_numpy()
        except KeyError:
            print(df)
        book_vectors.append(vectors)
    return book_vectors


def book_similarity(ratings_filtered, ratings_filtered_isbn, sim_df):
    sim_scores = []
    for book in ratings_filtered_isbn:
        sim_score2 = 0
        users = ratings_filtered[ratings_filtered["ISBN"] == book]["userID"].values
        for user in users:
            value = float(sim_df.loc[user].values[0])
            sim_score2 += value
        sim_scores.append(sim_score2)

    return sim_scores


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
            where=(intersection + union) != 0,
        )
        correlations.append(corr)

    total_correlation = np.sum(correlations, axis=0)
    return total_correlation


def crossover(bestMemdf, R, N):
    newpop = []
    columns = bestMemdf.iloc[:, :N]
    collist = columns.values.tolist()
    for _ in range(R):
        pair = random.sample(collist, 2)
        combined_books = list(set(pair[0] + pair[1]))
        if len(combined_books) >= N:
            children = random.sample(combined_books, N)
            if children not in newpop:
                newpop.append(children)
    return newpop


def similarityCal(book_sim, dic_sim, df2):
    final_array = []
    for column in df2.columns:
        book_series = df2[column]
        sc = np.where(np.isin(book_series, book_sim.index), book_series.map(dic_sim), 0)
        final_array.append(sc)

    final_array = np.array(final_array)
    sim_scores = np.sum(final_array, axis=0)

    return sim_scores


def calc_predic(book_series, ratings, targetUser, isbn_to_userid, ratings_filtered, sim_df):
    pr_scores = np.zeros(len(book_series))
    mean_tu = mean(ratings.loc[targetUser]["bookRating"])
    for idx, isbn in enumerate(book_series):
        users = isbn_to_userid.get(isbn, [])
        if not users:
            continue
        score = 0
        for u in users:
            rating_u_arr = ratings_filtered[
                (ratings_filtered['userID'] == u) & (ratings_filtered["ISBN"] == isbn)
            ]['bookRating'].values
            if len(rating_u_arr) == 0:
                continue
            rating_u = rating_u_arr[0]
            mean_u = sim_df.loc[int(u)]['mean']
            sim_u = sim_df.loc[int(u)][targetUser]
            if sim_u == 0:
                continue
            numerator = (rating_u - mean_u) * sim_u
            result = mean_tu + numerator / sim_u
            score += result
        pr_scores[idx] = score
    return pr_scores


def predict(df, ratings_filtered_isbn, ratings, targetUser, isbn_to_userid, ratings_filtered, sim_df):
    final_array = []

    for column in df.columns:
        book_series = df[column]
        array = np.where(np.isin(book_series, ratings_filtered_isbn), calc_predic(book_series, ratings, targetUser, isbn_to_userid, ratings_filtered, sim_df), 0)
        final_array.append(array)
    
    final_array = np.array(final_array)
    predict_scores = np.sum(final_array, axis=0)

    return predict_scores