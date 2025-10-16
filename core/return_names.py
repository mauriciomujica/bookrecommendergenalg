import pandas as pd
import os
import sys

try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    books_path = os.path.join(base_dir, "books_data/books_data_og", "books.csv")
    ratings_path = os.path.join(base_dir, "books_data/books_data_og", "ratings.csv")

    books_og = pd.read_csv(
        books_path,
        index_col="ISBN",
        delimiter=";",
        encoding="ISO-8859-1",
        on_bad_lines="skip",
        dtype={"Year-Of-Publication": str},
    ).sort_index()
    ratings = pd.read_csv(
        ratings_path,
        index_col="User-ID",
        delimiter=";",
        encoding="ISO-8859-1",
        on_bad_lines="skip",
    ).sort_index()
except FileNotFoundError:
    print("No se encuentran los datasets. Ejecutar download_data.py primero")
    print(base_dir)
    print(books_path)
    sys.exit()


def get_names(df, N):
    columns = df.iloc[:2, :N]
    collist = columns.values.tolist()
    nombres = {}
    for individuo in collist:
        for book in individuo:
            nombre = books_og.loc[book]["Book-Title"]
            nombres[book] = nombre
    return nombres
