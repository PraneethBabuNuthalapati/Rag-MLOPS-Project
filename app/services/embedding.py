import os
import warnings
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
cache_dir = project_root / ".cache" / "huggingface"
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(cache_dir))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    return model.encode(texts)
