import os

from dotenv import load_dotenv

load_dotenv()


def _get_env_var(env_var_str: str) -> str:
    env_var = os.getenv(env_var_str)

    if env_var is None:
        raise RuntimeError(f"Environment variable {env_var_str} is not set.")

    return env_var


API_URL = _get_env_var("API_URL")
