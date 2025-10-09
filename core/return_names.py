import pandas as pd
import sys

try:        
    books_og = pd.read_csv("books_data/books_data_og/books.csv", index_col="ISBN", delimiter=';', encoding="ISO-8859-1", on_bad_lines='skip', dtype={'Year-Of-Publication':str}).sort_index()
    ratings = pd.read_csv("books_data/books_data_og/ratings.csv", index_col="User-ID", delimiter= ';', encoding="ISO-8859-1", on_bad_lines='skip').sort_index()
except FileNotFoundError:
    print("No se encuentran los datasets. Ejecutar download_data.py primero")
    sys.exit()

def get_names(df, N):
    columns = df.iloc[:3, :N]
    collist = columns.values.tolist()
    for individuo in collist:
        for book in individuo:
            nombre = books_og.loc[book]['Book-Title']
            autor = books_og.loc[book]['Book-Author']
            print(f"{nombre} por {autor}")
        print("\n")