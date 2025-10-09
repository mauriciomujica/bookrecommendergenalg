import pandas as pd
import sys
import os

def get_names(books, base_dir):
    try:
        csv_path = os.path.join(base_dir, "../bookrecommendergenalg/books_data/books_data_og/books.csv")        
        books_og = pd.read_csv(csv_path, index_col="ISBN", delimiter=';', encoding="ISO-8859-1", on_bad_lines='skip', dtype={'Year-Of-Publication':str}).sort_index()
    except FileNotFoundError:
        print("No se encuentran los datasets. Ejecutar download_data.py primero")
        sys.exit()
    nombres = []
    for book in books:
        nombre = books_og.loc[book]['Book-Title']
        nombres.append(nombre)
    return nombres
