import pandas as pd
import gen_alg

if __name__ == "__main__":
    books = pd.read_csv("books_data/books.csv")
    ratings = pd.read_csv("books_data/ratings.csv")
    user = 277157  # userID
    M = 10000  # initial size of pop
    N = 10  # number of books inside of an individual
    R = 0.5  # ratio of which the newpop is generated using the initialpop size
    currentGen = 5
    maxGen = 10
    pop = gen_alg.initialPop(user, ratings, books, M, N)

    while currentGen != maxGen:
        correlations = gen_alg.correlationCal(pop, books)
        df = pd.DataFrame(list(zip(pop, correlations)), columns = ['Individual', 'Correlation Value'])
        df_sorted = df.sort_values(by = 'Correlation Value', ascending = False)
        bestMem = df_sorted.iloc[:round(len(df_sorted) * R)]
        newpop = gen_alg.crossover(bestMem)
        similarity = gen_alg.similarityCal(ratings, newpop, user)
        df2 = pd.DataFrame(list(zip(newpop, similarity)), columns = ['Individual', 'Similarity Value'])
        df2_sorted = df2.sort_values(by = 'Similarity Value', ascending = False)
        bestMem2 = df2_sorted.iloc[:round(len(df2_sorted) * R)]
        bestmem2_list = list(bestMem2['Individual'])

        pop = bestmem2_list
        currentGen += 1
    
    predict_scores = gen_alg.predict(ratings, bestmem2_list, user)
    df3 = pd.DataFrame(list(zip(bestmem2_list, predict_scores)), columns = ['Individual', 'Total Predicted Score'])
    bestMemfinal = df3.sort_values(by = 'Total Predicted Score', ascending = False)
    print(bestMemfinal)

