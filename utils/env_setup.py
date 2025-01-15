import os
from dotenv import load_dotenv


def setup_environment(required_vars=None) -> None:
    """
    Load environment variables and verify required variables are set
    :param required_vars: [List[str]], List of environment variable names that must be set.
                        If None, defaults to ['OPENAI_API_KEY']
    :raises: ValueError, if any required environment variable is not set
    """
    # Load environment variables from .env file if it exists (.env is used only for local execution)
    if os.path.exists('.env'):
        load_dotenv()
    
    # Default to checking OPENAI_API_KEY if no variables specified
    if required_vars is None:
        required_vars = ['OPENAI_API_KEY']
    
    # Check all required variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"The following required environment variables are not set: {', '.join(missing_vars)}\n"
            "Please set them in your .env file or environment variables."
        )

