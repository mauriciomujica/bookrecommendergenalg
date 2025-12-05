from setuptools import setup, find_packages

setup(
    name="bookrecommendergenalg",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "Flask==3.1.2",
        "flask-cors==6.0.1",
        "google-generativeai==0.8.5",
        "python-dotenv==1.0.0",
        "pandas==2.2.3",
        "gunicorn==21.2.0",
        "gdown==5.2.0"
    ],
    entry_points={
        "console_scripts": [
            "bookrecommendergenalg=bookrecommendergenalg.__main__:main"
        ]
    }
)