import pandas as pd
import gen_alg

if __name__ == "__main__":
    books = pd.read_csv("books_data/books.csv", index_col = "ISBN").sort_index()
    ratings = pd.read_csv("books_data/ratings.csv", index_col = "userID").sort_index()
    user = 277157  # userID
    M = 10000  # initial size of pop
    N = 10  # number of books inside of an individual
    S = 0.2
    R = 0.8  
    currentGen = 5
    maxGen = 10
    pop = gen_alg.initialPop(user, ratings, books, M, N)

    while currentGen != maxGen:
        correlations = gen_alg.correlationCal(pop, books)
        df = pd.DataFrame(list(zip(pop, correlations)), columns = ['Individual', 'Correlation Value'])
        df_sorted = df.sort_values(by = 'Correlation Value', ascending = False)
        bestMem = df_sorted.iloc[:round(len(df_sorted) * S)]
        newpop = gen_alg.crossover(bestMem, int(len(df) * R))
        similarity = gen_alg.similarityCal(ratings, newpop, user)
        df2 = pd.DataFrame(list(zip(newpop, similarity)), columns = ['Individual', 'Similarity Value'])
        df2_sorted = df2.sort_values(by = 'Similarity Value', ascending = False)
        bestMem2 = df2_sorted.iloc[:round(len(df2_sorted) * S)]
        nextgenpop = gen_alg.crossover(bestMem2, int(len(df) * R))

        pop = nextgenpop
        currentGen += 1
    
    final_mem = list(nextgenpop['Individual'])
    predict_scores = gen_alg.predict(ratings, final_mem, user)
    df3 = pd.DataFrame(list(zip(final_mem, predict_scores)), columns = ['Individual', 'Total Predicted Score'])
    bestMemfinal = df3.sort_values(by = 'Total Predicted Score', ascending = False)
    print(bestMemfinal)

