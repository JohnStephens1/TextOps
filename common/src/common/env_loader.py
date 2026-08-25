import logging
import os

import dotenv

from .logging_config import setup_logging

setup_logging()


logger = logging.getLogger()


class MissingEnvironmentVariableError(RuntimeError):
    pass


class MissingDotEnvFileError(RuntimeError):
    pass


def get_env_var(env_var_str: str) -> str:
    env_var = os.getenv(env_var_str)

    if env_var is not None:
        return env_var

    logger.info(
        f"Couldn't find environment variable. Trying to find {env_var_str} in dot env file..."
    )

    dot_env_file_path = dotenv.find_dotenv()

    if not dot_env_file_path:
        raise MissingDotEnvFileError(
            f"Neither environment variable {env_var_str} nor dot env file could be found."
        )

    env_values = dotenv.dotenv_values(dot_env_file_path)
    env_var = env_values.get(env_var_str)

    if env_var is None:
        raise MissingEnvironmentVariableError(
            f"Environment variable {env_var_str} could not be found in environment or dot env file."
        )

    return env_var
