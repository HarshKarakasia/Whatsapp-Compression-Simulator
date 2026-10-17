"""Source-grouped batch sampler for training (Phase 7). Each batch draws
`sources_per_batch` distinct source_ids and `views_per_source` files per
source, so every batch naturally contains the same-source positive pairs
L_con needs."""
import json
import random
from pathlib import Path

from src.data.split import group_by_source

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_real_manifest(path: Path | str | None = None) -> list[dict]:
    path = Path(path) if path else REPO_ROOT / "data_cache" / "manifest" / "metadata_real.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


class SourceGroupedBatcher:
    def __init__(self, rows: list[dict], sources_per_batch: int, views_per_source: int, seed: int = 0):
        self.groups = group_by_source(rows)
        self.source_ids = list(self.groups.keys())
        self.sources_per_batch = sources_per_batch
        self.views_per_source = views_per_source
        self.rng = random.Random(seed)

    def sample_batch(self) -> list[dict]:
        n = min(self.sources_per_batch, len(self.source_ids))
        chosen_sources = self.rng.sample(self.source_ids, k=n)

        batch = []
        for sid in chosen_sources:
            pool = self.groups[sid]
            k = self.views_per_source
            picks = self.rng.sample(pool, k=k) if len(pool) >= k else self.rng.choices(pool, k=k)
            batch.extend(picks)
        return batch
