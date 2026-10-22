# Lab 1 Docker development steps

Deploy `predict_monthly_salary.py` only. It takes years of experience and returns a monthly salary in INR. Leave `predict_exam_score.py` as a learning script until this container works.

The learning script trains, prints, and opens a plot with `plt.show()`. The container does not do that. It loads a saved model and answers HTTP requests.

Work from `lab1_linear_regression/`.

## 1. Save the fitted model

The API should not retrain the ten rows on every request. Fit once, write the model to disk, and copy that file into the image.

Create `train_salary_model.py`:

```python
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

years_of_experience = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    dtype=float,
).reshape(-1, 1)
monthly_salary_inr = np.array(
    [25000, 30000, 35000, 40000, 48000, 55000, 60000, 68000, 75000, 90000],
    dtype=float,
)

salary_model = LinearRegression()
salary_model.fit(years_of_experience, monthly_salary_inr)
joblib.dump(salary_model, "salary_model.joblib")
print("saved salary_model.joblib")
```

Run it with the project virtual environment:

```bash
python train_salary_model.py
```

`salary_model.joblib` is the artifact the image will load. Do not import matplotlib here.

## 2. Check a prediction locally before Docker

Confirm the saved model matches the script. Two years and five years are the same checks as `predict_monthly_salary.py`.

```bash
python -c "import joblib, numpy as np; m = joblib.load('salary_model.joblib'); print(m.predict(np.array([[2.0], [5.0]])))"
```

If those numbers disagree with the learning script, fix the training file before writing the API.

## 3. Write the API

Create `app.py`. It loads the model once at startup and exposes two routes.

```python
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load("salary_model.joblib")
app = FastAPI()


class Experience(BaseModel):
    years: float = Field(ge=0, le=50)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: Experience):
    salary = model.predict(np.array([[body.years]]))[0]
    return {"years": body.years, "monthly_salary_inr": round(float(salary))}
```

`GET /health` only checks that the process is up. `POST /predict` is the model. `years` must be between 0 and 50. The model expects shape `(n_samples, n_features)`, so one employee is `[[years]]`.

Run it on your machine before building an image:

```bash
pip install fastapi uvicorn joblib
uvicorn app:app --reload --port 8000
```

In another terminal:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"years": 5}'
```

`http://localhost:8000/docs` shows the same two routes in a browser. Stop this server before you rely on the container, or the port will already be taken.

## 4. Pin the image dependencies

Create `requirements.txt` in this lab folder. The root `requirements.txt` includes matplotlib, which this API does not use.

```text
scikit-learn==1.9.1
joblib==1.6.0
fastapi==0.161.0
uvicorn==0.54.0
```

Use the joblib version already installed in `.venv` if `1.6.0` does not match. Check with:

```bash
python -c "import joblib; print(joblib.__version__)"
```

Save and load the model with the same scikit-learn version. A model dumped by 1.9.1 should be loaded by 1.9.1.

## 5. Write the Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY salary_model.joblib app.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`python:3.12-slim` is the runtime. `WORKDIR /app` is where the later commands run. Dependencies install before the model and `app.py` are copied, so a code change does not reinstall packages. `--host 0.0.0.0` lets a request from your machine reach the process inside the container. `EXPOSE` documents the port. It does not publish it. The `docker run -p` flag does that.

## 6. Ignore files the image does not need

Create `.dockerignore`:

```text
__pycache__/
predict_monthly_salary.py
predict_exam_score.py
train_salary_model.py
```

The learning scripts and the training script stay on your machine. The image only needs `requirements.txt`, `salary_model.joblib`, and `app.py`.

## 7. Build the image

From `lab1_linear_regression/`:

```bash
docker build -t salary-api .
```

`-t salary-api` names the image. `.` is this folder, which is the build context. Docker can only `COPY` files from that context.

Rebuild after you change `app.py`, `Dockerfile`, or the saved model. A requirements change reruns the `pip install` layer. An `app.py` change reuses the installed packages.

## 8. Run the container

```bash
docker run --rm -p 8000:8000 salary-api
```

`--rm` deletes the container when it stops. `-p 8000:8000` maps your port 8000 to the container port 8000. Leave this terminal open while you test.

Call it the same way as the local server:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"years": 5}'
```

A successful predict response looks like:

```json
{"years": 5.0, "monthly_salary_inr": 55212}
```

The exact salary should match step 2. Stop the container with Ctrl+C.

## 9. Develop in a loop

1. Change `app.py`.
2. `docker build -t salary-api .`
3. `docker run --rm -p 8000:8000 salary-api`
4. `curl` the route you changed.

When the model data changes, rerun `python train_salary_model.py`, then build again. The image does not watch your files. `--reload` belongs on the local `uvicorn` command in step 3, not in the image `CMD`.

## Files in this folder when the loop works

| File | Where it runs |
| --- | --- |
| `predict_monthly_salary.py` | Your machine. Training plot. |
| `predict_exam_score.py` | Your machine. Second learning script. |
| `train_salary_model.py` | Your machine. Writes `salary_model.joblib`. |
| `salary_model.joblib` | Built locally, copied into the image. |
| `app.py` | Inside the container. |
| `requirements.txt` | Used by the image build. |
| `Dockerfile` | Build instructions. |
| `.dockerignore` | Keeps the learning scripts out of the image. |
