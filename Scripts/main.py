import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, preprocess_data, save_pipeline
from ml_functions import train_model, evaluate_model, save_model
from helper_functions import setup_logging, log_info, log_error

import pickle

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, os.getenv('DATA_DIR'))
print(DATA_DIR)
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))
LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(BASE_DIR, "logs"))


DATA_PATH = os.path.join(DATA_DIR,"raw","mobile_phone_price_prediction.csv")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessing_pipeline.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")


# os.makedirs(ARTIFACTS_DIR, exist_ok=True)
# os.makedirs(LOGS_DIR, exist_ok=True)

def main():
    try:
        log_info("🚀 Starting Mobile Price Prediction Pipeline")

        # Load data
        df = load_data(DATA_PATH)
        log_info(f"✅ Loaded dataset with shape: {df.shape}")

        # Preprocess data
        df_processed, pipeline = preprocess_data(df)
        log_info("✅ Data preprocessing completed")

        # Drop rows with NaN in target
        df_processed = df_processed.dropna(subset=["Price"])
        log_info(f"✅ Data shape after dropping NaN target: {df_processed.shape}")

        # Split features and target
        X = df_processed.drop("Price", axis=1)
        y = df_processed["Price"]

        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        log_info(f"✅ Split data: Train={X_train.shape[0]}, Val={X_val.shape[0]}")

        # Train model
        model = train_model(X_train, y_train)
        log_info("✅ Model training completed")

        # Evaluate
        mse, r2 = evaluate_model(model, X_val, y_val)
        log_info(f"📊 Evaluation - MSE: {mse:.4f}, R2: {r2:.4f}")

        # Save model, pipeline, features
        save_model(model, MODEL_PATH)
        log_info(f"✅ Model saved at: {MODEL_PATH}")

        save_pipeline(pipeline, PIPELINE_PATH)
        log_info(f"✅ Preprocessing pipeline saved at: {PIPELINE_PATH}")

        with open(FEATURE_NAMES_PATH, "wb") as f:
            pickle.dump(list(X.columns), f)
        log_info(f"✅ Feature names saved at: {FEATURE_NAMES_PATH}")

        log_info("🎉 Pipeline completed successfully!")

    except Exception as e:
        log_error(f"❌ Pipeline failed: {e}")
        import traceback
        log_error(traceback.format_exc())

if __name__ == "__main__":
    main()