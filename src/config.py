import os
import sys
from dotenv import load_dotenv

script_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from src.paths import PARENT_DIR
# from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig


# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'taxi_demand_calvin'
try:
    # HOPSWORKS_PROJECT_NAME = os.environ['HOPSWORKS_PROJECT_NAME']
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create an .env file on the project root with the HOPSWORKS_API_KEY')

# TODO: remove FEATURE_GROUP_NAME and FEATURE_GROUP_VERSION, and use FEATURE_GROUP_METADATA instead
FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1