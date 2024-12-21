import pandas as pd
import gen_alg
import ast
import heapq

if __name__ == "__main__":
    books = pd.read_csv("books_data/books.csv")
    ratings = pd.read_csv("books_data/ratings.csv")
    user = 277157  # userID
    M = 1000  # initial size of pop
    N = 10  # number of books inside of an individual
    R = 0.8  # ratio of which the newpop is generated using the initialpop
    currentGen = 5
    maxGen = 10
    pop = gen_alg.initialPop(user, ratings, books, M, N)

    while currentGen != maxGen:
        correlations = gen_alg.correlationCal(pop, books)
        d = {str(k): v for k, v in zip(pop, correlations)}
        d_sorted = {
            k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)
        }
        bestMem = list(d_sorted.keys())[0 : round(len(d_sorted) * R)]
        newpop = gen_alg.crossover(bestMem)
        similarity = gen_alg.similarityCal(ratings, newpop, user)
        d2 = {str(k): v for k, v in zip(newpop, similarity)}
        d_sorted2 = {
            k: v for k, v in sorted(d2.items(), key=lambda item: item[1], reverse=True)
        }
        bestMem2 = list(d_sorted2.keys())[0 : round(len(d_sorted2) * R)]
        pop = bestMem2
        currentGen += 1

print(d_sorted2)

best = list(map(list, d_sorted2.items()))

L = round(len(d_sorted) * R)

test = list(heapq.nlargest(L, d_sorted, key=d_sorted.get))
