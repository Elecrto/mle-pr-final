import numpy as np
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from fastapi.openapi.utils import get_openapi

from utils import (
    rec_store,
    preprocess_data,
    dedup_ids,
    describe_by_name,
    name_by_id,
    filter_ids,
    cat_cols,
    num_cols,
    mlc_model,
    kmeans_model,
    one_hot_drop,
    standart_scaler,
    kmeans_parquet,
    ncodpers_dict,
    products_catalog,
)
import json

app = FastAPI()
schema = get_openapi(title="My API", version="1.0.0", routes=app.routes)
json.dumps(schema, allow_nan=False)

load_dotenv()

logger = logging.getLogger("uvicorn.error")
logging.basicConfig(filename="rec_history.log", level=logging.INFO)
logging.info("Started")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загружает оффлайн-рекомендации при старте приложения
    """
    logging.info("Starting")

    for rec_type in ["personal", "classify", "default"]:
        columns = ["ncodpers", "product_name", "score"] if rec_type != "default" else ["product_name", "score"]
        rec_store.load(type=rec_type, columns=columns)

    yield
    logging.info("Stopping")


app = FastAPI(title="FastAPI-микросервис для выдачи рекомендаций", lifespan=lifespan)

# Инициализация метрик
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

METRIC = Counter(
    "http_requested_products_total",
    "Number of times a certain product has been requested.",
    labelnames=["recs_blended_describe_ru"],
)


@app.post("/recommendations", name="Получение рекомендаций для клиента")
async def recommendations(ncodpers_dict: dict = ncodpers_dict, k: int = 5):
    """
    Возвращает список k рекомендаций для клиента.
    Смешивает онлайн и оффлайн рекомендации.
    """
    logging.info("-------------------------------------")

    # Получение оффлайн и онлайн рекомендаций
    recs_offline = rec_store.get(ncodpers_dict["ncodpers"], k)
    recs_online_mlc = (await get_online_mlc(ncodpers_dict))["recs"]
    recs_online_kmeans = (await get_online_kmeans(ncodpers_dict))["recs"]

    logging.info(f"recs_offline: {describe_by_name(recs_offline)}")
    logging.info(f"recs_online_mlc: {describe_by_name(recs_online_mlc)}")
    logging.info(f"recs_online_kmeans: {describe_by_name(recs_online_kmeans)}")

    # Смешивание рекомендаций поочередно
    min_len = min(len(recs_offline), len(recs_online_mlc), len(recs_online_kmeans))
    recs_blended = sum(
        [[recs_online_mlc[i], recs_online_kmeans[i], recs_offline[i]] for i in range(min_len)], []
    )
    recs_blended += recs_offline[min_len:] + recs_online_mlc[min_len:] + recs_online_kmeans[min_len:]

    # Постобработка рекомендаций
    recs_blended = dedup_ids(recs_blended)
    recs_blended = filter_ids(ncodpers_dict["ncodpers"], recs_blended)[:k]
    recs_blended_describe_ru = describe_by_name(recs_blended)

    if recs_blended:
        logging.info(f"Итоговые рекомендации: {recs_blended_describe_ru}")

    # Логгируем метрики
    try:
        for rec_name in recs_blended_describe_ru:
            METRIC.labels(rec_name).inc()
    except Exception as e:
        print("RUNNING WITHOUT PROMETHEUS", e)

    return {"recs": recs_blended}


@app.post("/get_online_mlc")
async def get_online_mlc(ncodpers_dict: dict = ncodpers_dict) -> dict:
    """Возвращает онлайн-рекомендации от MLC модели"""
    logging.info("-------------------------------------")

    recs_id = []
    ncodpers_dict = preprocess_data(ncodpers_dict)

    if ncodpers_dict:
        features = [ncodpers_dict[col] for col in cat_cols + num_cols]
        recs = mlc_model.predict_proba(features)
        recs_scores = np.sort(recs)[::-1]
        recs_id = name_by_id(np.argsort(recs)[::-1])

        logging.info("Онлайн-рекомендации MLC:")
        for name, score in zip(describe_by_name(recs_id), recs_scores):
            logging.info(f"{name}, {score}")

    return {"recs": recs_id}


@app.post("/get_online_kmeans")
async def get_online_kmeans(ncodpers_dict: dict = ncodpers_dict) -> dict:
    """Возвращает онлайн-рекомендации от KMeans"""
    logging.info("-------------------------------------")

    recs_id = []
    ncodpers_dict = preprocess_data(ncodpers_dict)

    if ncodpers_dict:
        cat_vals = [ncodpers_dict[c] for c in cat_cols]
        num_vals = [ncodpers_dict[c] for c in num_cols]

        try:
            encoded_cat = one_hot_drop.transform([cat_vals])
            scaled_num = standart_scaler.transform([num_vals])
            features = np.concatenate([scaled_num[0], encoded_cat[0]])

            cluster_id = kmeans_model.predict([features])[0]
            recs = kmeans_parquet[kmeans_parquet["labels"] == cluster_id]
            recs_id = recs["product_name"].tolist()
            recs_scores = recs["score"].tolist()

            logging.info("Онлайн-рекомендации KMeans:")
            for name, score in zip(describe_by_name(recs_id), recs_scores):
                logging.info(f"{name}, {score}")

        except Exception as e:
            logging.warning(f"KMeans рекомендация невозможна: {e}")

    return {"recs": recs_id}


@app.get("/load_recommendations", name="Загрузка оффлайн-рекомендаций")
async def load_recommendations(rec_type: str = "classify"):
    """Перезагружает оффлайн-рекомендации из файла"""
    logging.info("-------------------------------------")
    columns = ["ncodpers", "product_name", "score"] if rec_type != "default" else ["product_name", "score"]
    rec_store.load(type=rec_type, columns=columns)


@app.get("/get_statistics", name="Статистика по рекомендациям")
async def get_statistics():
    """Возвращает статистику по рекомендациям"""
    logging.info("-------------------------------------")
    return rec_store.stats()

# Команда запуска:
# uvicorn recommendations_service:app --reload