from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
BROWSING_PROTECTED_TOP_K = 8
BUYING_PROTECTED_TOP_K = 4
NORMAL_RETRIEVAL_LIMIT = 50
RECOVERY_RETRIEVAL_LIMIT = 100
RECOVERY_START_TURN = 4
BASE_DIR = Path(__file__).resolve().parents[1]
DENSE_EMBEDDINGS_PATH = (
    BASE_DIR / "data" / "dense_embeddings.npy"
)
DENSE_ASINS_PATH = (
    BASE_DIR / "data" / "dense_asins.json"
)
LOCAL_DENSE_MODEL_PATH = (
    BASE_DIR / "models" / "all-MiniLM-L6-v2"
)
DENSE_MODEL_ID = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
DIVERSITY_START_TURN = 4
DIVERSITY_PROTECTED_TOP_K = int(
    os.getenv(
        "DIVERSITY_PROTECTED_TOP_K",
        "1",
    )
)
PROFILE_TIE_MARGIN = 0.03
ASK_ORDER = (
    "feature",
    "material",
    "color",
    "size",
    "style",
    "use_case",
    "budget",
    "brand",
    "other",
)
QUESTION_ATTRIBUTES = (
    "material",
    "color",
    "size",
    "style",
    "use_case",
)
ATTRIBUTE_TERMS = {
    "material": (
        "cotton",
        "polyester",
        "nylon",
        "leather",
        "wool",
        "silk",
        "spandex",
        "rayon",
        "denim",
        "linen",
        "suede",
        "fleece",
    ),
    "color": (
        "black",
        "white",
        "blue",
        "red",
        "pink",
        "green",
        "brown",
        "gray",
        "grey",
        "purple",
        "yellow",
        "orange",
        "navy",
        "beige",
        "gold",
        "silver",
    ),
    "size": (
        "small",
        "medium",
        "large",
        "xl",
        "xxl",
        "wide",
        "narrow",
        "petite",
        "plus size",
        "one size",
    ),
    "style": (
        "formal",
        "casual",
        "classic",
        "vintage",
        "slim",
        "loose",
        "oversized",
        "sleeveless",
        "long sleeve",
        "short sleeve",
        "v-neck",
        "crew neck",
        "hooded",
    ),
    "use_case": (
        "hiking",
        "running",
        "gym",
        "outdoor",
        "work",
        "winter",
        "travel",
        "walking",
        "sports",
        "wedding",
    ),
}

def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)

def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]

def _add_unique(items: list[str], value: str) -> None:
    value = value.strip(" .;,")
    if value and value not in items:
        items.append(value)

def _extract_no_preference_attribute(
    text: str,
) -> str | None:
    lowered = text.lower()
    attribute_pattern = (
        r"(material|color|size|style|brand|budget|"
        r"feature|use[_ -]?case|other)"
    )
    patterns = (
        # I don't have a preference for material.
        rf"(?:i\s+)?(?:don't|do not)\s+have\s+"
        rf"(?:an?\s+)?(?:additional\s+)?preference\s+"
        rf"(?:for|on|about)\s+{attribute_pattern}",
        # I have no preference for color.
        rf"(?:i\s+)?have\s+no\s+(?:additional\s+)?"
        rf"preference\s+(?:for|on|about)\s+"
        rf"{attribute_pattern}",
        # No preference on size.
        rf"no\s+(?:additional\s+)?preference\s+"
        rf"(?:for|on|about)\s+{attribute_pattern}",
        # I'm flexible on style.
        rf"(?:i(?:'m| am)\s+)?flexible\s+"
        rf"(?:on|about|with)\s+{attribute_pattern}",
        # Style doesn't matter to me.
        rf"{attribute_pattern}\s+"
        rf"(?:doesn't|does not)\s+matter"
        rf"(?:\s+to\s+me)?",
        # Anything is fine for brand.
        rf"anything\s+is\s+fine\s+"
        rf"(?:for|on|with)\s+{attribute_pattern}",
        # Use your judgment for material.
        # Also accepts British spelling: judgement.
        rf"use\s+your\s+judg(?:e)?ment\s+"
        rf"(?:for|on|with)\s+{attribute_pattern}",
    )
    for pattern in patterns:
        match = re.search(
            pattern,
            lowered,
            re.IGNORECASE,
        )
        if match:
            attribute = match.group(1)
            attribute = (
                attribute
                .replace("-", "_")
                .replace(" ", "_")
            )
            return attribute
    return None

def _extract_override_value(
    text: str,
) -> str | None:
    message = text.strip()
    patterns = (
        # Official evaluator:
        # Actually, ignore my earlier preference.
        # What I need is: cotton.
        r"what\s+i\s+need\s+is\s*:\s*(.+)$",
        # Instead, I need cotton.
        # Instead, I want cotton.
        r"\binstead\b\s*[,;:.]?\s*"
        r"(?:i\s+(?:need|want|would\s+like)\s+)"
        r"(.+)$",
        # Scratch that, I need cotton.
        # Scratch that, cotton.
        r"\bscratch\s+that\b\s*[,;:.]?\s*"
        r"(?:i\s+(?:need|want|would\s+like)\s+)?"
        r"(.+)$",
        # Forget that, I want cotton.
        r"\bforget\s+that\b\s*[,;:.]?\s*"
        r"(?:i\s+(?:need|want|would\s+like)\s+)?"
        r"(.+)$",
        # I'd rather have cotton.
        # I would rather have cotton.
        r"\bi(?:'d|\s+would)\s+rather\s+"
        r"(?:have|get|want)\s+(.+)$",
        # Change it to cotton.
        r"\bchange\s+it\s+to\s+(.+)$",
        # Actually, make it cotton.
        r"\bactually\b\s*[,;:.]?\s*"
        r"make\s+it\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(
            pattern,
            message,
            re.IGNORECASE,
        )
        if match:
            value = match.group(1).strip(
                " .,!?:;" )
            if value:
                return value
    return None

def _profile_question_attributes(
    user_profile: dict,
) -> set[str]:
    preferred_attributes: set[str] = set()
    if not isinstance(user_profile, dict):
        return preferred_attributes
    raw_tags = user_profile.get(
        "preference_tags",
        []
    )
    if not isinstance(raw_tags, list):
        return preferred_attributes
    for raw_tag in raw_tags:
        tag = str(raw_tag).lower().strip()
        if not tag:
            continue
        # Direct attribute interests
        if "material" in tag:
            preferred_attributes.add(
                "material"
            )
        if "color" in tag:
            preferred_attributes.add(
                "color"
            )
        if "size" in tag:
            preferred_attributes.add(
                "size"
            )
        if "style" in tag:
            preferred_attributes.add(
                "style"
            )
        # Fit can often be clarified through
        # size or style.
        if "fit" in tag:
            preferred_attributes.add(
                "size"
            )
            preferred_attributes.add(
                "style"
            )
        # Context-related historical interests.
        if (
            "warmth" in tag
            or "weather" in tag
        ):
            preferred_attributes.add(
                "use_case"
            )
    return preferred_attributes

def _classify_value(value: str) -> str:
    lowered = value.lower()
    #Budget
    if (
        "$" in lowered
        or "budget" in lowered
        or "under " in lowered
        or "<=" in lowered
    ):
        return "budget"
    #Material
    materials = (
        "cotton",
        "polyester",
        "nylon",
        "leather",
        "wool",
        "silk",
        "spandex",
        "rayon",
        "fabric",
    )
    if any(word in lowered for word in materials):
        return "material"
    #Color
    colors = (
        "black",
        "white",
        "blue",
        "red",
        "pink",
        "green",
        "brown",
        "gray",
        "grey",
        "purple",
        "yellow",
        "orange",
    )
    if any(word in lowered for word in colors):
        return "color"
    #Size
    if any(
        word in lowered
        for word in (
            "size",
            "sizing",
            "wide",
            "narrow",
            "width",
        )
    ):
        return "size"
    #Use case
    if any(
        word in lowered
        for word in (
            "hiking",
            "running",
            "gym",
            "outdoor",
            "work",
            "winter",
            "travel",
        )
    ):
        return "use_case"
    #Style
    if any(
        word in lowered
        for word in (
            "style",
            "fit",
            "formal",
            "casual",
            "sleeve",
            "neck",
        )
    ):
        return "style"
    return "feature"

class Agent:
    """Stateful hybrid conversational retrieval agent for TechJam Track 4."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self.product_texts: dict[str, str] = {}
        self.product_clusters: dict[str, int] = {}
        self._build_index()
        self._load_product_clusters()
        self.dense_enabled = False
        self.dense_model = None
        self.dense_embeddings = None
        self.dense_asins = []
        self._init_dense()

    def _init_dense(self) -> None:
        if os.getenv("DISABLE_DENSE") == "1":
            return
        if SentenceTransformer is None:
            return
        if (
            not DENSE_EMBEDDINGS_PATH.exists()
            or not DENSE_ASINS_PATH.exists()
        ):
            return
        try:
            dense_embeddings = np.load(
                DENSE_EMBEDDINGS_PATH
            )
            dense_asins = json.loads(
                DENSE_ASINS_PATH.read_text(
                    encoding="utf-8"
                )
            )
            if (
                len(dense_embeddings)
                != len(dense_asins)
            ):
                return
            # Preferred path:
            # use the bundled local MiniLM model.
            if LOCAL_DENSE_MODEL_PATH.exists():
                dense_model = SentenceTransformer(
                    str(LOCAL_DENSE_MODEL_PATH),
                    local_files_only=True,
                )
            # Optional development fallback:
            # only contact Hugging Face when explicitly allowed.
            elif os.getenv("ALLOW_HF_HUB") == "1":
                dense_model = SentenceTransformer(
                    DENSE_MODEL_ID
                )
            # Final safe fallback:
            # disable dense retrieval and use BM25.
            else:
                return
            self.dense_model = dense_model
            self.dense_embeddings = dense_embeddings
            self.dense_asins = dense_asins
            self.dense_enabled = True
        except Exception:
            self.dense_model = None
            self.dense_embeddings = None
            self.dense_asins = []
            self.dense_enabled = False

    def _load_product_clusters(
        self,
    ) -> None:
        cluster_path = Path(
            "data/product_clusters.json"
        )
        if not cluster_path.exists():
            return
        try:
            raw = json.loads(
                cluster_path.read_text(
                    encoding="utf-8"
                )
            )
            self.product_clusters = {
                str(asin): int(cluster_id)
                for asin, cluster_id
                in raw.items()
            }
        except Exception:
            self.product_clusters = {}

    def _diversify_results(
        self,
        candidates: list[str],
        top_k: int,
    ) -> list[str]:
        if (
            not self.product_clusters
            or len(candidates)
            <= DIVERSITY_PROTECTED_TOP_K
        ):
            return candidates[:top_k]
        protect_k = min(
            DIVERSITY_PROTECTED_TOP_K,
            top_k,
            len(candidates),
        )
        selected = list(
            candidates[:protect_k]
        )
        selected_set = set(
            selected
        )
        seen_clusters = set()
        for asin in selected:
            cluster_id = (
                self.product_clusters.get(
                    asin
                )
            )
            if cluster_id is not None:
                seen_clusters.add(
                    cluster_id
                )
        # First pass:
        # Prefer unseen semantic clusters.
        for asin in candidates[protect_k:]:
            if len(selected) >= top_k:
                break
            cluster_id = (
                self.product_clusters.get(
                    asin
                )
            )
            if (
                cluster_id is not None
                and cluster_id
                not in seen_clusters
            ):
                selected.append(
                    asin
                )
                selected_set.add(
                    asin
                )
                seen_clusters.add(
                    cluster_id
                )
        # Second pass:
        # Fill remaining positions using
        # the original ranking.
        if len(selected) < top_k:
            for asin in candidates[protect_k:]:
                if len(selected) >= top_k:
                    break
                if asin in selected_set:
                    continue
                selected.append(
                    asin
                )
                selected_set.add(
                    asin
                )
        return selected

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                search_text = " ".join([
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                ]).lower()
                self.product_texts[asin] = search_text
                batch.append(
                    (
                        asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "intent": "browsing",
            "category": "",
            "slots": {
                "material": [],
                "color": [],
                "size": [],
                "style": [],
                "use_case": [],
                "budget": [],
                "brand": [],
                "feature": [],
                "other": [],
            },
            "hard_constraints": [],
            "preference_log": [],
            "asked": [],
            "ignored": set(),
            "last_asked":None,
            "profile_question_attributes":
                _profile_question_attributes(
                    user_profile
                ),
        }

    def _update_slot(
        self,
        state: dict,
        slot: str,
        value: str,
        replace: bool = False,
    ) -> None:
        if slot not in state["slots"]:
            slot = "other"
        if replace:
            state["slots"][slot] = []
        _add_unique(
            state["slots"][slot],
            value,
        )

    def _add_preference(
        self,
        state: dict,
        slot: str,
        value: str,
    ) -> None:
        self._update_slot(
            state,
            slot,
            value,
            replace=False,
        )
        state["preference_log"].append({
            "slot": slot,
            "value": value,
            })

    def _remove_last_preference(
        self,
        state: dict,
    ) -> None:
        if not state["preference_log"]:
            return
        old = state["preference_log"].pop()
        slot = old["slot"]
        value = old["value"]
        if slot in state["slots"]:
            if value in state["slots"][slot]:
                state["slots"][slot].remove(value)

    def _update_state(self, state: dict, message: str) -> None:
        lowered = message.lower()
        #extract category from phrases.
        if not state["category"]:
            match = re.search(
                r"looking for\s+(.+?)(?:[.,]|$)",
                message,
                re.IGNORECASE,
            )
            if match:
                state["category"] = match.group(1).strip()
        #boundary/no-preference response
        attribute = _extract_no_preference_attribute(
            message
        )
        if attribute is not None:
            state["ignored"].add(
                attribute
            )
            if attribute in state["slots"]:
                state["slots"][attribute] = []
            state["preference_log"] = [
                item
                for item in state["preference_log"]
                if item.get("slot") != attribute
            ]
            if state["last_asked"] == attribute:
                state["last_asked"] = None
            return
        #Explicit hard constraint
        marker = "a key requirement is:"
        if marker in lowered:
            index = lowered.index(marker) + len(marker)
            value = message[index:].strip(" .")
            slot = _classify_value(value)
            self._update_slot(
                state,
                slot,
                value,
                replace=False,
            )
            _add_unique(state["hard_constraints"], value)
            state["intent"] = "buying"
            return
        #User changes previous preference
        override_value = _extract_override_value(message)
        if override_value is not None:
            self._remove_last_preference(
                state
            )
            slot = _classify_value(
                override_value
            )
            self._update_slot(
                state,
                slot,
                override_value,
                replace=False,
            )
            _add_unique(
                state["hard_constraints"],
                override_value,
            )
            state["intent"] = "buying"
            return
        #User answers one of our clarification questions
        marker = "for that, what matters is:"
        if marker in lowered:
            index = lowered.index(marker) + len(marker)
            value = message[index:].strip(" .")
            for part in value.split(";"):
                part = part.strip()
                if not part:
                    continue
                slot = state["last_asked"]
                if not slot:
                    slot = _classify_value(part)
                self._add_preference(
                    state,
                    slot,
                    part,
                )
            state["intent"] = "buying"
            return
        #Initial browsing message
        if "still exploring" in lowered:
            state["intent"] = "browsing"
            return
        #Initial soft preference after the category sentence
        if "looking for" in lowered and "." in message:
            trailing = message.split(".", 1)[1].strip(" .")
            if trailing:
                slot = _classify_value(trailing)
                self._add_preference(
                    state,
                    slot,
                    trailing,
                )

    def _build_query(self, state: dict) -> str:
        parts = []
        if state["category"]:
            parts.append(state["category"])
        parts.extend(state["hard_constraints"])
        for values in state["slots"].values():
            parts.extend(values)
        return " ".join(
            dict.fromkeys(parts)
        )

    def _retrieval_mode(
        self,
        state: dict,
    ) -> str:
        if state["hard_constraints"]:
            return "buying"
        return "browsing"

    def _dense_search(
            self,
            query: str,
            limit: int,
        ) -> list[str]:
            if (
                not query
                or not self.dense_enabled
                or self.dense_model is None
                or self.dense_embeddings is None
            ):
                return []
            try:
                query_embedding = self.dense_model.encode(
                    [query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )[0].astype(np.float32)
            except Exception:
                self.dense_enabled = False
                return []
            scores = (
                self.dense_embeddings
                @ query_embedding
            )
            limit = min(
                limit,
                len(scores),
            )
            candidate_indices = np.argpartition(
                -scores,
                limit - 1,
            )[:limit]
            candidate_indices = (
                candidate_indices[
                    np.argsort(
                        -scores[candidate_indices]
                    )
                ]
            )
            return [
                self.dense_asins[index]
                for index in candidate_indices
            ]

    def _candidate_attribute_values(
        self,
        asin: str,
        attribute: str,
    ) -> set[str]:
        text = self.product_texts.get(
            asin,
            "",
        )
        terms = ATTRIBUTE_TERMS.get(
            attribute,
            (),
        )
        return {
            term
            for term in terms
            if term in text
        }

    def _question_value(
        self,
        candidates: list[str],
        attribute: str,
    ) -> float:
        if not candidates:
            return 0.0
        products_with_value = 0
        unique_values: set[str] = set()
        for asin in candidates:
            values = self._candidate_attribute_values(
                asin,
                attribute,
            )
            if values:
                products_with_value += 1
                unique_values.update(values)
        coverage = (
            products_with_value
            / len(candidates)
        )
        diversity = min(
            len(unique_values) / 4.0,
            1.0,
        )
        return coverage * diversity

    def _next_question(
        self,
        state: dict,
        candidates: list[str],
        turn: int,
    ) -> str | None:
        if turn <= 3:
            for attribute in ASK_ORDER:
                if (
                    attribute not in state["asked"]
                    and attribute not in state["ignored"]
                ):
                    state["asked"].append(
                        attribute
                    )
                    state["last_asked"] = (
                        attribute
                    )
                    return attribute
        available = [
            attribute
            for attribute in QUESTION_ATTRIBUTES
            if (
                attribute not in state["asked"]
                and attribute not in state["ignored"]
            )
        ]
        attribute_scores: dict[str, float] = {}
        for attribute in available:
            score = self._question_value(
                candidates,
                attribute,
            )
            attribute_scores[attribute] = score
        best_attribute = None
        best_score = 0.0
        if attribute_scores:
            best_attribute = max(
                attribute_scores,
                key=attribute_scores.get,
            )
            best_score = attribute_scores[
                best_attribute
            ]
            profile_attributes = state.get(
                "profile_question_attributes",
                set(),
            )
            # Only let long-term profile influence
            # genuinely near-tied clarification choices.
            near_tied_attributes = [
                attribute
                for attribute, score
                in attribute_scores.items()
                if (
                    best_score - score
                    <= PROFILE_TIE_MARGIN
                )
            ]
            for attribute in near_tied_attributes:
                if attribute in profile_attributes:
                    best_attribute = attribute
                    best_score = attribute_scores[
                        attribute
                    ]
                    break
        # Only trust candidate-aware selection
        # when the signal is reasonably useful.
        if (
            best_attribute is not None
            and best_score >= 0.20
        ):
            state["asked"].append(
                best_attribute
            )
            state["last_asked"] = (
                best_attribute
            )
            return best_attribute
        # Fallback to our original stable order.
        for attribute in ASK_ORDER:
            if (
                attribute not in state["asked"]
                and attribute not in state["ignored"]
            ):
                state["asked"].append(
                    attribute
                )
                state["last_asked"] = (
                    attribute
                )
                return attribute
        state["last_asked"] = None
        return None

    def _bm25_search(
        self,
        query: str,
        limit: int,
    ) -> list[str]:
        unique_terms = list(
            dict.fromkeys(_terms(query))
        )[:40]
        if not unique_terms:
            return []
        expression = " OR ".join(
            f'"{term}"'
            for term in unique_terms
        )
        rows = self.connection.execute(
            "SELECT parent_asin FROM products "
            "WHERE products MATCH ? "
            "ORDER BY bm25("
            "products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"
            ") LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [
            str(row[0])
            for row in rows
        ]

    def _weighted_rrf(
        self,
        bm25_results: list[str],
        dense_results: list[str],
    ) -> list[str]:
        scores: dict[str, float] = {}
        for rank, asin in enumerate(
            bm25_results,
            start=1,
        ):
            scores[asin] = (
                scores.get(asin, 0.0)
                + 1.0 / (60 + rank)
            )
        for rank, asin in enumerate(
            dense_results,
            start=1,
        ):
            scores[asin] = (
                scores.get(asin, 0.0)
                + 0.10 / (60 + rank)
            )
        return [
            asin
            for asin, _ in sorted(
                scores.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ]

    def _constraint_rerank(
        self,
        candidates: list[str],
        state: dict,
    ) -> list[str]:
        if (
            not state["hard_constraints"]
            and not state["preference_log"]
        ):
            return candidates
        scored = []
        for original_rank, asin in enumerate(
            candidates,
            start=1,
        ):
            text = self.product_texts.get(
                asin,
                "",
            )
            hard_matches = 0
            soft_matches = 0
            for constraint in state["hard_constraints"]:
                terms = _terms(constraint)
                if terms and all(
                    term in text
                    for term in terms
                ):
                    hard_matches += 1
            for preference in state["preference_log"]:
                value = preference["value"]
                terms = _terms(value)
                if terms and all(
                    term in text
                    for term in terms
                ):
                    soft_matches += 1
            scored.append(
                (
                    asin,
                    hard_matches,
                    soft_matches,
                    original_rank,
                )
            )
        scored.sort(
            key=lambda item: (
                -item[1],
                -item[2],
                item[3],
            )
        )
        return [
            asin
            for asin, _, _, _ in scored
        ]

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        self._update_state(
            state,
            user_message,
        )
        query = self._build_query(state)
        retrieval_mode = self._retrieval_mode(
            state
        )
        if turn >= RECOVERY_START_TURN:
            retrieval_limit = RECOVERY_RETRIEVAL_LIMIT
        else:
            retrieval_limit = NORMAL_RETRIEVAL_LIMIT
        if retrieval_mode == "buying":
            protected_top_k = BUYING_PROTECTED_TOP_K
        else:
            protected_top_k = BROWSING_PROTECTED_TOP_K
        bm25_results = self._bm25_search(
            query,
            retrieval_limit,
        )
        dense_results = self._dense_search(
            query,
            retrieval_limit,
        )
        protected = bm25_results[:protected_top_k]
        bm25_tail = bm25_results[protected_top_k:]
        if dense_results:
            fused_tail = self._weighted_rrf(
                bm25_tail,
                dense_results,
            )
        else:
            fused_tail = bm25_tail
        fused_tail = [
            asin
            for asin in fused_tail
            if asin not in protected
        ]
        final_results = (
            protected
            + fused_tail
        )
        final_results = self._constraint_rerank(
            final_results,
            state,
        )
        if (
            retrieval_mode == "browsing"
            and turn >= DIVERSITY_START_TURN
        ):
            recommendation_asins = (
                self._diversify_results(
                    final_results,
                    top_k,
                )
            )
        else:
            recommendation_asins = (
                final_results[:top_k]
            )
        recommendations = [
            {"parent_asin": asin}
            for asin in recommendation_asins
        ]
        question_candidates = (
            final_results[:20]
        )
        ask_attribute = self._next_question(
            state,
            question_candidates,
            turn,
        )
        if ask_attribute:
            response_message = (
                f"Here are my current matches. "
                f"Do you have a {ask_attribute} preference?"
            )
        else:
            response_message = (
                "Here are the closest matches based on your preferences."
            )
        return {
            "message": response_message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }