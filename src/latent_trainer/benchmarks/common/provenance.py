import importlib.metadata
import subprocess
from pathlib import Path


def software_provenance() -> dict[str, str | None]:
    root = Path(__file__).resolve().parents[5]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages: dict[str, str | None] = {}
    for name in ("latent-trainer", "sc2-datasets", "torch", "scikit-learn", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"git_commit": commit, **packages}
