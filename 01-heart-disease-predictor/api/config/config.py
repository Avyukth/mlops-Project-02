import os
import tomllib
from pathlib import Path

from dotenv import dotenv_values


class Config:
    def __init__(self, config_path="config.toml", env_path=".env"):
        self.config_path = Path(__file__).resolve().parent.parent / config_path
        self.env_path = Path(__file__).resolve().parent.parent / env_path
        self._config_data = {}
        self.load_config()

    def load_config(self):
        print(f"Attempting to load config from: {self.config_path.resolve()}")
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {self.config_path.resolve()}"
            )

        # Load .env file
        env_vars = dotenv_values(self.env_path)
        self._config_data.update(env_vars)

        # Load TOML config
        try:
            with open(self.config_path, "rb") as config_file:
                toml_data = tomllib.load(config_file)
            self._config_data.update(toml_data)
        except tomllib.TOMLDecodeError as e:
            print(f"Error parsing TOML: {str(e)}")
            raise

        # Set attributes
        for key, value in self._config_data.items():
            setattr(self, key.upper(), value)

        print("Config loaded successfully")

    def __getitem__(self, key):
        return self._config_data.get(key)

    def get(self, key, default=None):
        return self._config_data.get(key, default)

    def __str__(self):
        config_str = "Loaded Configuration:\n"
        for key, value in self._config_data.items():
            if not key.startswith("_"):  # Exclude private attributes
                config_str += f"{key.upper()}: {value}\n"
        return config_str
