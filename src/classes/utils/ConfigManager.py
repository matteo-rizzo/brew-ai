import yaml


class ConfigManager:
    """
    A class to manage configuration loading and validation.
    """

    def __init__(self, config_file="config.yml", verbose=False):
        self.config_file = config_file
        self.verbose = verbose
        self.config = self._load_config()

    def _load_config(self):
        """
        Load configurations from a YAML file.

        :return: A dictionary containing configurations.
        """
        try:
            if self.verbose:
                print(f"Loading configuration from {self.config_file}...")
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                if not isinstance(config, dict):
                    raise ValueError("Configuration file must contain a valid YAML object.")
                if self.verbose:
                    print("Configuration successfully loaded.")
                return config
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error while loading configuration file: {e}")

    def validate(self):
        """
        Validate the configuration dictionary to ensure all required keys are present.

        :raises ValueError: If required keys are missing.
        """
        config = self.config
        required_keys = ["model_name", "model_file"]
        if not any(key in config for key in ["input_file", "db_url"]):
            required_keys.extend(["input_file", "db_url", "db_input_table"])
        if not any(key in config for key in ["output_file", "db_output_table"]):
            required_keys.extend(["output_file", "db_output_table"])

        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
        if self.verbose:
            print("Configuration validation successful.")
