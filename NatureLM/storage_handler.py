from pathlib import Path
from typing import Union

from cloudpathlib import GSPath as CloudGSPath
from NatureLM.storage_utils import is_gcs_path

class StorageHandler:
    def __init__(self, backend: str = "local"):
        assert backend in ["local", "gcs"], "backend must be 'local' or 'gcs'"
        self.backend = backend

    # def resolve(self, path: Union[str, Path]) -> Union[Path, CloudGSPath]:
    #     path = str(path)
    #     if self.backend == "gcs":
    #         if not is_gcs_path(path):
    #             raise ValueError(f"Expected gs:// path for GCS backend, got: {path}")
    #         return CloudGSPath(path)
    #     return Path(path).expanduser().resolve()
    def resolve(self, path: str | Path | None):
        if path is None:
            return None
        path = str(path)

        # gcs backend unchanged
        if self.backend == "gcs":
            if path.startswith("gs://"):
                return CloudGSPath(path)
            raise ValueError("Expected gs://...")
        
        # local backend: if it exists on disk → Path, else return string
        p = Path(path).expanduser()
        return p.resolve() if p.exists() or p.is_absolute() else path
