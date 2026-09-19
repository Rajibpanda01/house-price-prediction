from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "house_price_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURES_PATH = ARTIFACT_DIR / "feature_profile.json"

REPORTS_DIR = ROOT_DIR / "reports"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

TARGET = "SalePrice"
ID_COLUMN = "Id"
RANDOM_STATE = 42
