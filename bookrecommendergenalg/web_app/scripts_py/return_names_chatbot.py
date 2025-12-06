import pandas as pd
import sys
import os


def get_names(books, base_dir):
    try:
        csv_path = os.path.join(
            base_dir, "../bookrecommendergenalg/books_data/books_info.csv"
        )
        books_og = pd.read_csv(
            csv_path,
            index_col="ISBN",
            dtype={"Year-Of-Publication": str},
        ).sort_index()
    except FileNotFoundError:
        print("No se encuentran los datasets. Ejecutar download_data.py primero")
        sys.exit()
    nombres = []
    for book in books:
        row = books_og.loc[book]
        nombres.append({
            "isbn": book,
            "title": row["Book-Title"],
            "author": row["Book-Author"],
            "image_url": row["Image-URL-L"]
        })
    return nombres
