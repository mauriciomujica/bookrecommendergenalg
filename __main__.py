import argparse
from bookrecommendergenalg.cli_app.main import main as run_cli
from bookrecommendergenalg.web_app.app import start_server

def main():
    parser = argparse.ArgumentParser(description="Book Recommender Genetic Algorithm")
    parser.add_argument("--web", action="store_true", help="Abrir web app en lugar de CLI")
    parser.add_argument("--port", type=int, default=5000, help="Port para el servidor web")
    args = parser.parse_args()

    if args.web:
        start_server(port=args.port)
    else:
        run_cli()

if __name__ == "__main__":
    main()
