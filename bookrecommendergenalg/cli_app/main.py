import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from ..core import gen_alg, return_names


def main():
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        books_path = os.path.join(base_dir, "books_data", "books.csv")
        ratings_path = os.path.join(base_dir, "books_data", "ratings.csv")
        books = pd.read_csv(books_path, index_col="ISBN").sort_index()
        ratings = pd.read_csv(ratings_path, index_col="userID").sort_index()
    except FileNotFoundError:
        print("No se encuentran los datasets. Ejecutar download_data.py primero")
        print(books_path)
        sys.exit()

    targetUser = 277157  # userID
    rated_items = ratings.loc[targetUser]["ISBN"].tolist()
    similar_users = np.unique(
        np.concatenate(
            [ratings.index[ratings["ISBN"] == book].values for book in rated_items]
        )
    )
    similar_users = np.delete(similar_users, similar_users == targetUser)

    M = 10000
    N = 5
    S = 0.2
    R = 0.8
    currentGen = 0
    maxGen = 10

    sim_df = pd.DataFrame(index=similar_users, columns=[targetUser], dtype=float)
    sim_users = gen_alg.sim_matrix(
        targetUser, rated_items, ratings, similar_users, sim_df
    )

    ratings_filtered = ratings.loc[ratings.index.isin(sim_users.index)].reset_index()
    ratings_filtered_isbn = ratings_filtered["ISBN"].to_numpy()
    ratings_sim_score = gen_alg.book_similarity(
        ratings_filtered, ratings_filtered_isbn, sim_df
    )
    means = gen_alg.mean_users(similar_users, ratings_filtered)
    sim_users["mean"] = means

    book_sim = pd.DataFrame(
        list(zip(ratings_filtered_isbn, ratings_sim_score)),
        columns=["ISBN", "sim_score"],
    ).set_index("ISBN")
    dic_sim = dict(zip(ratings_filtered_isbn, ratings_sim_score))
    isbn_to_userid = defaultdict(list)
    for isbn, userid in zip(ratings_filtered["ISBN"], ratings_filtered["userID"]):
        isbn_to_userid[isbn].append(userid)

    pop = gen_alg.initialPop(rated_items, books, M, N)

    while currentGen != maxGen:
        df = pd.DataFrame(pop, columns=[f"Book_{i + 1}" for i in range(N)])
        book_vectors = gen_alg.get_vectors(df, books)
        correlations = gen_alg.correlationCal(book_vectors, N)
        df["Correlation Value"] = correlations
        df_sorted = df.sort_values(by="Correlation Value", ascending=False)
        bestMem = df_sorted.iloc[: round(len(df_sorted) * S)]
        newpop = gen_alg.crossover(bestMem, int(len(df) * R), N)
        df2 = pd.DataFrame(newpop, columns=[f"Book_{i + 1}" for i in range(N)])
        sim_scores = gen_alg.similarityCal(book_sim, dic_sim, df2)
        df2["Similarity Value"] = sim_scores.tolist()
        df2_sorted = df2.sort_values(by="Similarity Value", ascending=False)
        bestMem2 = df2_sorted.iloc[: round(len(df2_sorted) * S)]
        nextgenpop = gen_alg.crossover(bestMem2, int(len(df) * R), N)

        pop = nextgenpop
        print(f"Generación {currentGen + 1} completada.\n")
        currentGen += 1

    final_mem = bestMem2
    df3 = final_mem.copy()
    df3.drop(["Similarity Value"], axis=1, inplace=True)
    pr = gen_alg.predict(
        df3,
        ratings_filtered_isbn,
        ratings,
        targetUser,
        isbn_to_userid,
        ratings_filtered,
        sim_df,
    )

    df3["Predict Score"] = pr.tolist()
    bestMemfinal = df3.sort_values(by="Predict Score", ascending=False)
    names = return_names.get_names(bestMemfinal, N)
    print(names)


if __name__ == "__main__":
    main()
