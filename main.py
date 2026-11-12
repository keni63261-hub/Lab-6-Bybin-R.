from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model_utils import predict_iris

app = FastAPI(title="Iris Classifier")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """Отображение пустой формы"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None}
    )


@app.post("/", response_class=HTMLResponse)
async def predict(
        request: Request,
        sepal_length: float = Form(...),
        sepal_width: float = Form(...),
        petal_length: float = Form(...),
        petal_width: float = Form(...)
):
    """Обработка формы и вывод результата"""
    features = [sepal_length, sepal_width, petal_length, petal_width]

    try:
        prediction = predict_iris(features)
        error = None
    except Exception as e:
        prediction = None
        error = str(e)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": prediction,
            "error": error,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
    )