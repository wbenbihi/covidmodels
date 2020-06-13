import os
import pandas as pd
from datetime import datetime
from functools import reduce
from libs.azure_blob_connector import AzureBlobConnector
from libs.predictor import CovidPredictor
from utils.func_utils import isevaluatable, transform_group

# Load ENV Variables from .env
from dotenv import load_dotenv

load_dotenv()
# Logging
import logging

logging.basicConfig(level=logging.WARNING)
logger_azure = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger_azure.setLevel(logging.WARNING)

# Check ENV VARIABLES
assert os.environ.get("CREDENTIALS") is not None, "<CREDENTIALS> variable is unset"
assert os.environ.get("STORAGE_NAME") is not None, "<STORAGE_NAME> variable is unset"
assert (
    os.environ.get("SOURCE_FILENAME") is not None
), "<SOURCE_FILENAME> variable is unset"
assert (
    os.environ.get("DESTINATION_FILENAME") is not None
), "<DESTINATION_FILENAME> variable is unset"
assert (
    os.environ.get("KPI_OF_INTEREST") is not None
), "<KPI_OF_INTEREST> variable is unset"
assert isevaluatable(
    os.environ.get("KPI_OF_INTEREST")
), "<KPI_OF_INTEREST> not parsable as a <list>"
assert isevaluatable(os.environ.get("MULTIPROCESS")), "<MULTIPROCESS> not parsable"
assert isevaluatable(
    os.environ.get("VALIDATION_STEPS")
), "<VALIDATION_STEPS> not parsable"
assert isevaluatable(
    os.environ.get("TRAIN_DATA_POINTS")
), "<TRAIN_DATA_POINTS> not parsable"
assert isevaluatable(
    os.environ.get("STEPS_TO_PREDICT")
), "<STEPS_TO_PREDICT> not parsable"
assert isevaluatable(os.environ.get("CLIP_TO_ZERO")), "<CLIP_TO_ZERO> not parsable"

MULTIPROCESS = eval(os.environ.get("MULTIPROCESS", "True"))
HIERARCHICAL = eval(os.environ.get("HIERARCHICAL", "False"))
VALIDATION_STEPS = eval(os.environ.get("VALIDATION_STEPS", "7"))
TRAIN_DATA_POINTS = eval(os.environ.get("TRAIN_DATA_POINTS", "23"))
STEPS_TO_PREDICT = eval(os.environ.get("STEPS_TO_PREDICT", "0"))
CLIP_TO_ZERO = eval(os.environ.get("CLIP_TO_ZERO", "False"))
KPI_OF_INTEREST = eval(os.environ.get("KPI_OF_INTEREST"))


# Instantiate ABC
abc = AzureBlobConnector(
    credentials=os.environ.get("CREDENTIALS"),
    storage_name=os.environ.get("STORAGE_NAME"),
)
# Retrieve last available file as DataFrame
file_to_retrieve = abc.get_last_filename_version(os.environ.get("SOURCE_FILENAME"))
source_df = abc.open_as_dataframe(file_to_retrieve, sep=";", parse_dates=["jour"])

logging.basicConfig(level=logging.INFO)
predictors = []
for kpi in KPI_OF_INTEREST:
    logging.info(f"Covid Predictor {kpi}")
    p = CovidPredictor(
        df=source_df,
        kpi=kpi,
        multiprocess=MULTIPROCESS,
        is_hierarchical=HIERARCHICAL,
        validation_steps=VALIDATION_STEPS,
        steps_to_predict=STEPS_TO_PREDICT,
        clip_to_zero=CLIP_TO_ZERO,
        train_data_points=TRAIN_DATA_POINTS,
    )._generate_time_series()
    p()
    predictors.append(p)

prediction_df = reduce(
    lambda df1, df2: df1.join(df2),
    [
        p.get_results().set_index(["ds", "area"]).add_prefix(f"{p.kpi}_")
        for p in predictors
    ],
).reset_index()

groups = prediction_df.groupby('area')
prediction_df = groups.apply(lambda group: transform_group(group))

logger_azure = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger_azure.setLevel(logging.INFO)
# Send File
logging.info(f"FILENAME = {os.environ.get('DESTINATION_FILENAME')}{datetime.now().strftime('%Y-%m-%d')}.csv")
abc.send_data(
    buff=prediction_df.to_csv(sep=";", index=False),
    cloudpath=f"{os.environ.get('DESTINATION_FILENAME')}{datetime.now().strftime('%Y-%m-%d')}.csv",
)
