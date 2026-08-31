from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path("data/catalog.jsonl")
EMBEDDING_PATH = Path(
    "data/dense_embeddings.npy"
)
ASIN_PATH = Path(
    "data/dense_asins.json"
)
MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{key} {item}"
            for key, item in value.items()
        )
    if isinstance(value, list):
        return " ".join(
            str(item)
            for item in value
        )
    return str(value)

def product_text(product: dict) -> str:
    parts = [
        f"title: {_text(product.get('title'))}",
        f"categories: {_text(product.get('categories'))}",
        f"features: {_text(product.get('features'))}",
        f"details: {_text(product.get('details'))}",
        f"store: {_text(product.get('store'))}",
        f"description: {_text(product.get('description'))}",
    ]
    return " ".join(parts)

def main() -> None:
    texts = []
    asins = []
    with CATALOG_PATH.open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            product = json.loads(line)
            asins.append(
                str(product["parent_asin"])
            )
            texts.append(
                product_text(product)
            )
    print(
        f"Loaded {len(texts)} products."
    )
    print(
        f"Loading model: {MODEL_NAME}"
    )
    model = SentenceTransformer(
        MODEL_NAME
    )
    print(
        "Encoding product catalog..."
    )
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        )
    embeddings = embeddings.astype(
        np.float32
    )
    np.save(
        EMBEDDING_PATH,
        embeddings,
    )
    ASIN_PATH.write_text(
        json.dumps(asins),
        encoding="utf-8",
    )
    print(
        "Dense index saved."
        )
    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

if __name__ == "__main__":
    main()