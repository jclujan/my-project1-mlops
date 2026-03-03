import os


def ensure_parent_dir(file_path: str) -> None:
    """
    Create parent directory if it does not exist.
    Prevents FileNotFoundError when saving files.
    """
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)