from fastapi import FastAPI, File, UploadFile
from models.classifier import classify_image


app = FastAPI()


@app.get("/")
async def healthcheck():
    return "I am alive!"


@app.post("/classify")
async def classify(image: UploadFile = File(...)):
    # contents = await image.read()
    results = classify_image(image)
    return results

