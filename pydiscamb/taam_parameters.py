from pathlib import Path
from typing import List


def get_TAAM_root() -> Path:
    # Check if editable
    project_root = Path(__file__).parent.parent
    # Assume the existance of a couple files in the project
    # means that we are in editable
    if (
        (project_root / "CMakeLists.txt").exists()
        and (project_root / "pyproject.toml").exists()
        and (project_root / "README.md").exists()
        and (project_root / ".gitignore").exists()
    ):
        return project_root / "data"
    return Path(__file__).parent / "data"


def get_TAAM_databanks() -> List[str]:
    """
    Get a list of all available databanks.
    """
    # Upon installation, all *databank.txt in the data-folder
    # are copied into the installation directory of the module,
    # without preserving other folder structure.
    # Assume no filenames are duplicate.
    files = get_TAAM_root().glob("**/*databank.txt")
    return [str(file) for file in files]


def is_MATTS_installed() -> bool:
    return any("MATTS" in path for path in get_TAAM_databanks())


def get_default_databank() -> str:
    banks = get_TAAM_databanks()
    search = "MATTS" if is_MATTS_installed() else "default"
    for bank in banks:
        if search in bank:
            return bank
    # Failsafe
    return bank
