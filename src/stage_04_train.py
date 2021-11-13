import logging
import argparse
from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import load_model, get_unique_path_to_save_model 
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import os

stage = "stage_04"

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
logging.basicConfig(filename=os.path.join(logs_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    base_model_dir = artifacts["BASE_MODEL_DIR"]
    #base_model = os.path.join(artifacts_dir, base_model_dir, artifacts["BASE_MODEL_NAME"])

    updated_base_model = os.path.join(artifacts_dir, base_model_dir, artifacts["UPDATED_BASE_MODEL_NAME"])
    model = load_model(updated_base_model)

    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    train_generator, valid_generator = train_valid_generator(
        data_dir = artifacts["DATA_DIR"],
        IMAGE_SIZE = tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE = params["BATCH_SIZE"],
        do_data_augmentation = params["AUGMENTATION"]
    )

    steps_per_epochs = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    model.fit(
        train_generator,
        validation_data = valid_generator,
        epochs = params["EPOCHS"],
        steps_per_epoch = steps_per_epochs,
        validation_steps = validation_steps,
        callbacks = callbacks
    )
    logging.info("training completed")

    trained_model_dir_path = os.path.join(artifacts, artifacts["TRAINED_MODEL_DIR"])
    create_directory([trained_model_dir_path])

    model_file_path = get_unique_path_to_save_model(trained_model_dir_path)
    model.save(model_file_path)
    logging.info(f"trained model is saved at {model_file_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(f"<<<<< Stage {stage} started")
        train(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f"Stage {stage} completed, model trained and saved")
    except Exception as e:
        logging.exception(e)
        raise e