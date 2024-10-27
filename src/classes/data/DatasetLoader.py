import json
import os
from typing import Tuple, Optional, Dict

import openml
import pandas as pd

from src.classes.utils.Logger import Logger
from src.config import SUITE_ID, BENCHMARK


class DatasetLoader:
    def __init__(self, suite_id: int = SUITE_ID, benchmark: bool = BENCHMARK):
        """
        Initialize the DatasetLoader with suite ID and benchmark flag.

        :param suite_id: int, The OpenML suite ID to search within.
        :param benchmark: bool, Whether to retrieve configuration from a benchmark suite or the default config.
        """
        self.suite_id = str(suite_id)
        self.benchmark = benchmark
        self.logger = Logger()

    def get_config_path(self) -> str:
        """
        Determines the configuration file path based on the suite ID and benchmark flag.

        :return: str, Path to the configuration file.
        """
        suite_config_paths = {
            "334": 'benchmark/cat_clf_config.json',
            "335": 'benchmark/cat_reg_config.json',
            "336": 'benchmark/num_reg_config.json',
            "337": 'benchmark/num_clf_config.json'
        }
        return suite_config_paths.get(self.suite_id, 'dataset/config.json') if self.benchmark else 'dataset/config.json'

    def get_dataset_config(self, dataset_id: str) -> Dict:
        """
        Retrieve the dataset configuration based on the dataset ID, suite ID, and benchmark flag.

        :param dataset_id: str, The ID of the dataset to retrieve the configuration for.
        :return: dict, Configuration dictionary for the specified dataset.
        """
        config_path = self.get_config_path()

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if dataset_id in config:
                    return config[dataset_id]
                else:
                    self.logger.warning(f"Dataset ID '{dataset_id}' not found in {config_path}.")
                    raise KeyError(f"Dataset ID '{dataset_id}' not found in {config_path}.")
        except FileNotFoundError:
            self.logger.error(f"Configuration file '{config_path}' not found.")
            raise
        except KeyError as e:
            self.logger.error(f"Error accessing dataset ID '{dataset_id}': {e}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON in configuration file '{config_path}'.")
            raise

    def load_from_open_ml(self, dataset_name: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Load a dataset from OpenML by its name within a specified suite.

        :param dataset_name: str, The name of the dataset to load.
        :return: tuple (X, y) where X is the feature DataFrame and y is the target Series.
                 Returns None if the dataset is not found.
        """
        dataset_name = dataset_name.lower()
        try:
            benchmark_suite = openml.study.get_suite(int(self.suite_id))
            for task_id in benchmark_suite.tasks:
                task = openml.tasks.get_task(task_id)
                dataset = task.get_dataset()

                if dataset.name.lower() == dataset_name:
                    self.logger.info(f"Dataset '{dataset_name}' found in OpenML suite '{self.suite_id}'. Loading...")
                    x, y, _, _ = dataset.get_data(
                        dataset_format="dataframe", target=dataset.default_target_attribute
                    )
                    return x, y

            self.logger.warning(f"Dataset '{dataset_name}' not found in suite ID {self.suite_id}.")
            return None

        except Exception as e:
            self.logger.error(f"Error loading dataset from OpenML: {e}")
            raise

    def load_from_disk(self, dataset_id: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the dataset from a local CSV file defined in the settings.

        :param dataset_id: str, The ID of the dataset to load.
        :return: A tuple containing the feature DataFrame (X) and target Series (y).
        """
        path_to_data = os.path.join("dataset", f"{dataset_id}.csv")

        try:
            config = self.get_dataset_config(dataset_id)
            target = config["target"]
            df = pd.read_csv(path_to_data, index_col=False)
            x = df.drop(columns=[target, "Unnamed: 0"], errors='ignore')
            y = df[target]

            self.logger.info(f"Dataset '{dataset_id}' loaded successfully from disk.")
            return x, y

        except FileNotFoundError:
            self.logger.error(f"Dataset file '{path_to_data}' not found.")
            raise
        except KeyError as e:
            self.logger.error(f"Error accessing target column in dataset config for '{dataset_id}': {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset from '{path_to_data}': {e}")
            raise

    def load_dataset(self, dataset_id: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Load the dataset either from OpenML (if benchmark) or from local disk.

        :param dataset_id: str, The ID or name of the dataset to load.
        :return: A tuple (X, y) where X is the feature DataFrame and y is the target Series.
        """
        if self.benchmark:
            return self.load_from_open_ml(dataset_id)
        else:
            return self.load_from_disk(dataset_id)
