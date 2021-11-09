import yaml
import logging
import os
import time

def read_yaml(path_to_yaml:str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file {path_to_yaml} loade successfully")
    return content

def create_directory(dirs:list):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        logging.info(f"directory is created at {dir}")

def get_timestamp(name):
    timestamp = time.asctime().replace(" ","_").replace(":","_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name