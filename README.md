# Covid19 Prophet Model

# Setup
## Env Var
```bash
# 
STORAGE_NAME="<STORAGE_NAME>"
CREDENTIALS="<AZURE_STORAGE_CONNECTION_STRING>"
SOURCE_FILENAME="formated-covid-data-from-datagouvfr-" # Prefix of the source file
DESTINATION_FILENAME="formated-covid-predictions-" # Prefix of the destination file
# 
KPI_OF_INTEREST="['new_hosp', 'new_death']" # KPIs to forecast
MULTIPROCESS=True # Toggle Parallel processing
HIERARCHICAL=False # Whether or not to predict at country level
# 
TIME_KEY='jour' # Date Column in source file
AREA_KEY='dep' # Area Column in source file
VALIDATION_STEPS=0 # Validation steps (test only)
TRAIN_DATA_POINTS=23 # Numner of data points to use for training
STEPS_TO_PREDICT=10 # Steps to forecast
CLIP_TO_ZERO=False # Clip forecasted values to to zeros
```