import pandas as pd
import numpy as np
import gen_alg

if __name__ == "__main__":
    books = pd.read_csv("books_data/books.csv", index_col="ISBN").sort_index()
    ratings = pd.read_csv("books_data/ratings.csv", index_col="userID").sort_index()
    targetUser = 277157  # userID
    rated_items = ratings.loc[targetUser]["ISBN"].tolist()
    similar_users = np.unique(
        np.concatenate(
            [ratings.index[ratings["ISBN"] == book].values for book in rated_items]
        )
    )
    similar_users = np.delete(similar_users, similar_users == targetUser)

    M = 10000  # initial size of pop
    N = 5  # number of books inside of an individual
    S = 0.2
    R = 0.8
    currentGen = 0
    maxGen = 10

    sim_df = pd.DataFrame(index=similar_users, columns=[targetUser], dtype=float)
    sim_users = gen_alg.sim_matrix(
        targetUser, rated_items, ratings, similar_users, sim_df
    )
    ratings_filtered = ratings.loc[ratings.index.isin(sim_users.index)]
    ratings_filtered_grouped = ratings_filtered.reset_index().groupby(['ISBN', 'userID'])
    ratings_filtered_grouped_df = ratings_filtered_grouped.first()
    ratings_filtered_grouped_df = ratings_filtered_grouped_df.reset_index().set_index(['ISBN', 'userID'])
    c = ratings_filtered_grouped_df.index.get_level_values(0)
    c = np.array(c)
    pop = gen_alg.initialPop(rated_items, books, M, N)

    while currentGen != maxGen:
        df = pd.DataFrame(pop, columns=[f'Book_{i+1}' for i in range(N)])
        
        column_1 = df['Book_1']
        column_2 = df['Book_2']
        column_3 = df['Book_3']
        column_4 = df['Book_4']
        column_5 = df['Book_5']

        books1 = gen_alg.get_vectors(column_1, books)
        books2 = gen_alg.get_vectors(column_2, books)
        books3 = gen_alg.get_vectors(column_3, books)
        books4 = gen_alg.get_vectors(column_4, books)
        books5 = gen_alg.get_vectors(column_5, books)


        correlations = gen_alg.correlationCal([books1, books2, books3, books4, books5], N)
        df['Correlation Value'] = correlations
        df_sorted = df.sort_values(by="Correlation Value", ascending=False)
        bestMem = df_sorted.iloc[: round(len(df_sorted) * S)]
        newpop = gen_alg.crossover(bestMem, int(len(df) * R))

        
        similarity = gen_alg.similarityCal(ratings_filtered_grouped_df, newpop, sim_users, c)
        df2 = pd.DataFrame(
            list(zip(newpop, similarity)), columns=["Individual", "Similarity Value"]
        )
        df2_sorted = df2.sort_values(by="Similarity Value", ascending=False)
        bestMem2 = df2_sorted.iloc[: round(len(df2_sorted) * S)]
        nextgenpop = gen_alg.crossover(bestMem2, int(len(df) * R))

        pop = nextgenpop
        currentGen += 1

    final_mem = bestMem2
    predict_scores = gen_alg.predict(ratings, sim_users, final_mem, targetUser)
    df3 = pd.DataFrame(
        {
            "Individual": list(final_mem["Individual"]),
            "Total Predicted Score": predict_scores,
        }
    )
    bestMemfinal = df3.sort_values(by="Total Predicted Score", ascending=False)
    print(bestMemfinal)