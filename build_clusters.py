import json
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

EMBEDDINGS_PATH = Path(
    "data/dense_embeddings.npy"
)
ASINS_PATH = Path(
    "data/dense_asins.json"
)
OUTPUT_PATH = Path(
    "data/product_clusters.json"
)
N_CLUSTERS = 128

def main() -> None:
    print("Loading dense embeddings...")
    embeddings = np.load(
        EMBEDDINGS_PATH
    )
    asins = json.loads(
        ASINS_PATH.read_text(
            encoding="utf-8"
        )
    )
    if len(embeddings) != len(asins):
        raise ValueError(
            "Embeddings and ASIN count mismatch."
        )
    print(
        f"Products: {len(asins)}"
    )
    print(
        f"Embedding shape: {embeddings.shape}"
    )
    print(
        f"Training MiniBatchKMeans "
        f"with {N_CLUSTERS} clusters..."
    )
    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=2048,
        random_state=42,
        n_init="auto",
    )
    cluster_ids = model.fit_predict(
        embeddings
    )
    product_clusters = {
        asin: int(cluster_id)
        for asin, cluster_id
        in zip(
            asins,
            cluster_ids,
        )
    }
    OUTPUT_PATH.write_text(
        json.dumps(
            product_clusters,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    unique_clusters = len(
        set(cluster_ids.tolist())
    )
    print(
        f"Created {unique_clusters} clusters."
    )
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

if __name__ == "__main__":
    main()