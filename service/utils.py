import os
import io
import json
import pickle
import logging

import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
import boto3

# Загрузка переменных окружения
load_dotenv()

# Логгер
logger = logging.getLogger("uvicorn.error")

# Подключение к S3
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

session = boto3.session.Session()
s3 = session.client(
    service_name="s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Колонки
cat_cols = [
    "sexo", "ind_nuevo", "indext", "canal_entrada",
    "cod_prov", "ind_actividad_cliente", "segmento"
]
num_cols = ["age", "antiguedad", "renta"]

# Загрузка модели MLC
mlc_model = CatBoostClassifier(
    cat_features=cat_cols,
    loss_function="MultiLogloss",
    iterations=10,
    learning_rate=1,
    depth=2,
)
obj_mlc_model = s3.get_object(Bucket=BUCKET_NAME, Key=os.environ.get("MLC_MODEL"))
mlc_model = mlc_model.load_model(stream=io.BytesIO(obj_mlc_model["Body"].read()))

# Загрузка K-means модели и трансформеров
def load_pickle(key):
    return pickle.load(io.BytesIO(s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()))

kmeans_model = load_pickle(os.environ.get("KEY_KMEANS_MODEL"))
one_hot_drop = load_pickle(os.environ.get("KEY_KMEANS_MODEL_ONE_HOT_DROP"))
standart_scaler = load_pickle(os.environ.get("KEY_KMEANS_MODEL_STANDART_SCALER"))

# Загрузка датафреймов
def load_parquet(key):
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()))

kmeans_parquet = load_parquet(os.environ.get("KEY_KMEANS_PARQUET"))
products_catalog = load_parquet(os.environ.get("KEY_PRODUCTS_CATALOG_PARQUET"))
last_activity = load_parquet(os.environ.get("KEY_LAST_ACTIVITY_PARQUET"))

# Загрузка тестового клиента
obj_test_user_json = s3.get_object(Bucket=BUCKET_NAME, Key=os.environ.get("KEY_TEST_USER_JSON"))
ncodpers_dict = json.load(io.BytesIO(obj_test_user_json["Body"].read()))


# Класс для хранения и получения оффлайн-рекомендаций
class Recommendations:
    def __init__(self):
        self._recs = {"personal": None, "classify": None, "default": None}
        self._stats = {
            "request_personal_count": 0,
            "request_classify_count": 0,
            "request_default_count": 0,
        }

    def load(self, type, **kwargs):
        logger.info(f"Loading recommendations, type: {type}")

        if type == "personal":
            key = os.environ.get("KEY_PERSONAL_ALS_PARQUET")
            df = load_parquet(key).set_index("ncodpers")
        elif type == "classify":
            key = os.environ.get("KEY_MLC_PARQUET")
            df = load_parquet(key).set_index("ncodpers")
        else:
            key = os.environ.get("KEY_KMEANS_PARQUET")
            df = load_parquet(key)

        self._recs[type] = df
        logger.info("Loaded")

    def get(self, user_id: int, k: int = 5):
        recs = []
        found_type = None

        # Поиск персональных рекомендаций
        if self._recs["personal"] is not None:
            try:
                recs = (
                    self._recs["personal"].loc[user_id]
                    .sort_values(by="score", ascending=False)["product_name"]
                    .to_list()[:k]
                )
                self._stats["request_personal_count"] += 1
                found_type = "personal"
                logger.info(f"Found {len(recs)} personal recommendations!")
            except KeyError:
                pass

        # Поиск классификаторских рекомендаций
        if not recs and self._recs["classify"] is not None:
            try:
                recs = (
                    self._recs["classify"].loc[user_id]
                    .sort_values(by="score", ascending=False)["product_name"]
                    .to_list()[:k]
                )
                self._stats["request_classify_count"] += 1
                found_type = "classify"
                logger.info(f"Found {len(recs)} classify recommendations!")
            except KeyError:
                pass

        # Поиск топ-рекомендаций
        if not recs and self._recs["default"] is not None:
            recs = (
                self._recs["default"]
                .drop_duplicates(subset=["product_name"], keep="first")
                .sort_values(by="score", ascending=False)["product_name"]
                .to_list()[:k]
            )
            self._stats["request_default_count"] += 1
            found_type = "default"
            logger.info(f"Found {len(recs)} TOP-recommendations!")

        if not recs:
            logger.error("No recommendations found")

        return recs

    def stats(self):
        logger.info("Stats for recommendations:")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value}")
        print(self._stats)
        return self._stats


# Инициализация
rec_store = Recommendations()


# =======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================

def preprocess_data(ncodpers_dict: dict):
    """
    Предобработка входных данных клиента.
    """

    if pd.isnull(ncodpers_dict.get("sexo")):
        print("No sexo! No recommendations!")
        return []

    try:
        ncodpers_dict["ind_nuevo"] = int(ncodpers_dict["ind_nuevo"])
        ncodpers_dict["ind_actividad_cliente"] = int(ncodpers_dict["ind_actividad_cliente"])
        ncodpers_dict["indext"] = ncodpers_dict.get("indext") or "N"
        ncodpers_dict["canal_entrada"] = ncodpers_dict.get("canal_entrada") or "NAN"

        try:
            ncodpers_dict["cod_prov"] = int(ncodpers_dict["cod_prov"])
        except (ValueError, TypeError):
            ncodpers_dict["cod_prov"] = 0

        ncodpers_dict["segmento"] = ncodpers_dict.get("segmento") or "NAN"
        ncodpers_dict["antiguedad"] = max(0, int(ncodpers_dict["antiguedad"]))

        try:
            renta = int(ncodpers_dict["renta"])
            ncodpers_dict["renta"] = min(renta, 350000)
        except (ValueError, TypeError):
            ncodpers_dict["renta"] = 66964.6

    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return []

    return ncodpers_dict


def describe_by_name(names):
    """
    Возвращает описания продуктов по их short-name.
    """
    return [
        products_catalog[products_catalog["product_name"] == x]["description"].values[0]
        for x in names if not products_catalog[products_catalog["product_name"] == x].empty
    ]


def name_by_id(id_list):
    """
    Возвращает short-name продуктов по их индексам.
    """
    return [products_catalog.iloc[x]["product_name"] for x in id_list]


def filter_ids(ncodpers, names):
    """
    Исключает из списка те продукты, которые уже были у клиента в предыдущем месяце.
    """
    active = last_activity[last_activity["ncodpers"] == ncodpers]["product_name"].tolist()
    return [x for x in names if x not in active]


def dedup_ids(ids):
    """
    Удаляет дубликаты из списка, сохраняя порядок.
    """
    seen = set()
    return [x for x in ids if x not in seen and not seen.add(x)]
