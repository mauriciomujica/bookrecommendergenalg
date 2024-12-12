import pandas as pd
import gen_alg
import random


if __name__ == "__main__":
    books = pd.read_csv("books_data/books.csv")
    ratings = pd.read_csv("books_data/ratings.csv")
    user = 277157
    M = 10  # Cantidad de individuos a generar
    N = 5  # Cantidad de items dentro de los individuos
    currentGen = 0
    maxGen = 10
    topX = 5
    pop = gen_alg.initialPop(user, ratings, books, M, N)

    while currentGen != maxGen:
        correlations = gen_alg.correlationCal(pop, books)
        d = {str(k): v for k, v in zip(pop, correlations)}
        d_sorted = {
            k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)
        }
        bestMem = list(d_sorted.keys())[0:topX]
        newpop = gen_alg.crossover(bestMem)

        currentGen += 1


parents = random.choices(bestMem, k=2)


for i in bestMem:
    pair = random.choices(bestMem, k=2)
    print(pair)
