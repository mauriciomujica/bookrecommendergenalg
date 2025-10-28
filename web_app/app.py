import os
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
from bookrecommendergenalg.web_app.scripts_py import return_names_chatbot, web_to_alg

load_dotenv()
app = Flask(__name__)
# Esto es importante para permitir solicitudes desde la pagina web local
CORS(app, origins="*")

key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=key)

contexto = """(Responde en formato HTML, sin escribir ```html´´´). Eres un asistente de recomendador de lecturas, como un librero experimentado que te conoce mediante los gustos de los libros que lee. Tu función es ayudar a los usuarios respondiendo a sus preguntas de manera breve clara y concisa, justifica tu respuesta."""

# Puedemos mantener el historial de conversacion en una sesion.
# Solo para este ejemplo, usaremos una variable global.
mensajes = [{"role": "system", "content": contexto}]


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/index.html")
def index():
    return send_from_directory(".", "index.html")


@app.route("/pages/nosotros.html")
def nosotros():
    return send_from_directory("pages", "nosotros.html")


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory("js", filename)


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory("css", filename)


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory("assets", filename)


@app.route("/ask", methods=["POST"])
def ask_chatbot():
    # Obtiene el mensaje del usuario del cuerpo de la solicitud JSON
    data = request.get_json()
    pregunta = data.get("mensaje")

    if not pregunta:
        return jsonify({"error": "No se recibió un mensaje"}), 400

    # Agrega el mensaje del usuario al historial
    mensajes.append({"role": "user", "content": pregunta})

    # Construye el historial para Gemini
    historial = contexto + "\n"
    for mensaje in mensajes[1:]:
        if mensaje["role"] == "user":
            historial += f"Usuario: {mensaje['content']}\n"
        elif mensaje["role"] == "assistant":
            historial += f"Asistente: {mensaje['content']}\n"

    # Usa Gemini para generar la respuesta
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(historial)
        respuesta = response.text
        mensajes.append({"role": "assistant", "content": respuesta})
        return jsonify({"respuesta": respuesta})
    except Exception as e:
        # En caso de error, devuelve un mensaje de error
        return jsonify({"error": str(e)}), 500


@app.route("/search-csv", methods=["GET"])
def search_csv():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        # Determina dinámicamente la ruta absoluta al CSV
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csv_path = os.path.join(base_dir, "books_data", "ratings.csv")

        df = pd.read_csv(csv_path, index_col="userID").sort_index()
        rated_items = df.loc[int(user_id)]["ISBN"].tolist()

        nombres = return_names_chatbot.get_names(rated_items, base_dir)
        return jsonify({"results": nombres})
    except FileNotFoundError:
        return jsonify({"error": "CSV file not found"}), 500
    except pd.errors.EmptyDataError:
        return jsonify({"error": "CSV file is empty"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/selected", methods=["POST"])
def receive_selected():
    """Receive a JSON payload with a 'selected' list and 'userid'.
    Returns a simple confirmation JSON. This endpoint is intentionally lightweight;
    you can extend it to perform further actions (store to DB, trigger jobs, etc.).
    """
    try:
        data = request.get_json()
        if not data or "selected" not in data or "userid" not in data:
            return jsonify(
                {"error": "Request JSON must include 'selected' list and 'userid'"}
            ), 400

        selected = data.get("selected")
        userid = data.get("userid")

        if not isinstance(selected, list):
            return jsonify({"error": "'selected' must be a list"}), 400

        # Basic sanitization: ensure items are strings
        cleaned = [str(x) for x in selected]

        # Use userid in the processing logic
        bestMem = web_to_alg.main(cleaned, userid)

        # Generate synopses for each recommended book using Gemini API
        synopses = []
        synopsis_prompt = "Genera una breve sinopsis en español (máximo 3 oraciones) para el siguiente libro. Responde solo con la sinopsis, sin títulos ni formato adicional:"

        try:
            model = genai.GenerativeModel("gemini-2.5-flash")

            # bestMem is a dictionary with ISBN as key and title as value
            for isbn, book_title in bestMem.items():
                # Generate synopsis for each book using the title
                full_prompt = f"{synopsis_prompt} {book_title}"
                response = model.generate_content(full_prompt)
                synopsis = response.text.strip()

                # Add synopsis to the book data
                synopses.append(
                    {"isbn": isbn, "title": book_title, "synopsis": synopsis}
                )

        except Exception as e:
            # If synopsis generation fails, return books without synopses
            print(f"Error generating synopses: {str(e)}")
            # Convert dictionary to list format for fallback
            synopses = [
                {"isbn": isbn, "title": title} for isbn, title in bestMem.items()
            ]

        # Return results with synopses
        return jsonify({"results": synopses})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ejecuta el servidor en el puerto 5000

    app.run(debug=True, port=5000)
