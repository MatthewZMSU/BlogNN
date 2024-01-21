from fastapi import FastAPI

from .scripts.text_transforms import get_features
from .scripts.blog_prediction import model_predict


# model:

app = FastAPI()


# @app.on_event("startup")
# def startup_event():


@app.get("/")
def index():
    return {
        "message": "This is ML server. Please see the README.md for more information"
    }


@app.post("/text-transform")
def text_transform(text: str):
    return get_features(text)


@app.post("/predict")
def predict(features: list[float]):
    max_prob, matthew_prob = model_predict(features)
    author = 'Maxim' if max_prob >= matthew_prob else 'Matthew'
    message = ("Probabilities:\n"
               f"\t{max_prob * 100:.2f}% - Maxim's text\n"
               f"\t{matthew_prob * 100:.2f}% - Matthew's text\n\n"
               f"So I think that this text is written by: {author}")
    return {
        'message': message
    }
