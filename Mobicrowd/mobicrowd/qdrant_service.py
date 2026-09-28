# from typing import Optional, Any, Dict
#
# import requests
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams
# from qdrant_client.http.models import PayloadSchemaType
#
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.U-X7SiCR0ddi4atTpTDsfXIdqb2FqWYSNnJfQYADLwM"
# QDRANT_URL = "https://a3c5ba39-eab5-436f-8cd4-e17e800a5a11.eu-west-2-0.aws.cloud.qdrant.io"
# QDRANT_PORT = 6333
# QDRANT_COLLECTION = "photo_embeddings"
# QDRANT_COLLECTION_VIDEO = "video_embeddings"
# # Initialize the client
# client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY,
# )
#
# def create_photo_embeddings_collection():
#     client.recreate_collection(
#         collection_name=QDRANT_COLLECTION,  # e.g., "photo_embeddings"
#         vectors_config=VectorParams(
#             size=512,                # CLIP vector dimension
#             distance=Distance.COSINE
#         )
#     )
# # create_photo_embeddings_collection()
# def create_video_embeddings_collection():
#     client.recreate_collection(
#         collection_name=QDRANT_COLLECTION_VIDEO,  # e.g., "photo_embeddings"
#         vectors_config=VectorParams(
#             size=512,                # CLIP vector dimension
#             distance=Distance.COSINE
#         )
#     )
# # create_video_embeddings_collection()
#
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import PayloadSchemaType
#
# def create_index():
#     client.create_payload_index(
#         collection_name=QDRANT_COLLECTION,
#         field_name="photo_id",
#         field_schema=PayloadSchemaType.INTEGER
#     )
#     client.create_payload_index(
#         collection_name=QDRANT_COLLECTION,
#         field_name="event_id",
#         field_schema=PayloadSchemaType.INTEGER
#     )
# # create_index()
# def create_video_index():
#     client.create_payload_index(
#         collection_name=QDRANT_COLLECTION_VIDEO,
#         field_name="video_id",
#         field_schema=PayloadSchemaType.INTEGER
#     )
#     client.create_payload_index(
#         collection_name=QDRANT_COLLECTION_VIDEO,
#         field_name="event_id",
#         field_schema=PayloadSchemaType.INTEGER
#     )
# #     client.delete_collection(collection_name=QDRANT_COLLECTION)
# #
# #
# # create_video_index()
# # print("❌ Collection deleted.")
# # print("✅ Qdrant schema set successfully.")
#
#
# def qdrant_rest_call(
#     method: str,
#     path: str,
#     *,
#     payload: Optional[Dict[str, Any]] = None,
#     timeout: int = 20,
# ) -> Dict[str, Any]:
#     """
#     Generic Qdrant REST API caller.
#     - method: "GET" / "POST" / "PUT" / "DELETE"
#     - path: e.g. "/collections/photo_embeddings/points/search"
#     - payload: request JSON for POST/PUT
#     """
#     base = f"{QDRANT_URL}:{QDRANT_PORT}".rstrip("/")
#     url = base + (path if path.startswith("/") else f"/{path}")
#
#     headers = {
#         "api-key": QDRANT_API_KEY,
#         "Content-Type": "application/json",
#     }
#
#     resp = requests.request(
#         method=method.upper(),
#         url=url,
#         headers=headers,
#         json=payload,
#         timeout=timeout,
#     )
#
#     # Raise informative error
#     if not resp.ok:
#         try:
#             err_json = resp.json()
#         except Exception:
#             err_json = {"raw": resp.text}
#
#         raise RuntimeError(
#             f"Qdrant REST error {resp.status_code} {resp.reason}\n"
#             f"URL: {url}\n"
#             f"Response: {err_json}"
#         )
#
#     # Return JSON (or empty dict)
#     if resp.text.strip():
#         return resp.json()
#     return {}
#
# def qdrant_search_points(
#     *,
#     vector: list[float],
#     limit: int = 10,
#     event_id: Optional[int] = None,
# ) -> Dict[str, Any]:
#     """
#     Calls:
#       POST /collections/{collection}/points/search
#     with optional payload filter (event_id).
#     """
#     search_payload: Dict[str, Any] = {
#         "vector": vector,
#         "limit": limit,
#         "with_payload": True,
#         "with_vector": False,
#     }
#
#     if event_id is not None:
#         search_payload["filter"] = {
#             "must": [
#                 {"key": "event_id", "match": {"value": int(event_id)}}
#             ]
#         }
#
#     return qdrant_rest_call(
#         "POST",
#         f"/collections/{QDRANT_COLLECTION}/points/search",
#         payload=search_payload,
#     )
# def qdrant_healthcheck() -> Dict[str, Any]:
#     """
#     If this fails with 403 => API key is missing/wrong OR not being sent.
#     """
#     return qdrant_rest_call("GET", "/collections")
# # print(qdrant_healthcheck())
# # # Example: vector MUST be length 512 (your CLIP size)
# # dummy_vector = [0.0] * 512
# #
# # res = qdrant_search_points(vector=dummy_vector, limit=3, event_id=12)
# # print(res)
import os
import threading
from typing import Optional, Any, Dict
from urllib.parse import urlsplit, urlunsplit

import requests
from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import PayloadSchemaType


def _setting(name: str, default=None):
    """
    Read from Django settings first, then process environment, then default.
    """
    value = getattr(settings, name, None)
    if value not in (None, ""):
        return value

    value = os.environ.get(name)
    if value not in (None, ""):
        return value

    return default


QDRANT_API_KEY = _setting("QDRANT_API_KEY")
QDRANT_URL = str(_setting("QDRANT_URL", "")).rstrip("/")
QDRANT_PORT = int(_setting("QDRANT_PORT", 6333))

QDRANT_COLLECTION = _setting("QDRANT_COLLECTION", "photo_embeddings")
QDRANT_COLLECTION_VIDEO = _setting(
    "QDRANT_COLLECTION_VIDEO",
    "video_embeddings",
)
QDRANT_COLLECTION_TEXT = _setting(
    "QDRANT_COLLECTION_TEXT",
    "text_embeddings",
)

if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL is not configured.")

if not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_API_KEY is not configured.")


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# ---------------------------------------------------------------------
# Internal schema helpers
# ---------------------------------------------------------------------

_index_lock = threading.Lock()
_ready_indexes: set[tuple[str, str]] = set()


def _collection_exists(collection_name: str) -> bool:
    """
    Compatibility-friendly collection existence check.
    """
    collections = client.get_collections().collections
    return any(item.name == collection_name for item in collections)


def _payload_schema_type(field_info) -> Optional[str]:
    """
    Normalize the Qdrant payload schema representation.

    Depending on qdrant-client version, field_info can be a dict or a
    pydantic object.
    """
    if field_info is None:
        return None

    if isinstance(field_info, dict):
        value = field_info.get("data_type")
    else:
        value = getattr(field_info, "data_type", None)

    if value is None:
        return None

    value = getattr(value, "value", value)
    return str(value).lower()


def ensure_payload_index(
    *,
    collection_name: str,
    field_name: str,
    field_schema=PayloadSchemaType.INTEGER,
) -> None:
    """
    Ensure one payload index exists with the expected type.

    Safe to call repeatedly.

    This function NEVER recreates or deletes a collection and NEVER deletes
    existing points. If several Django workers race to create the same index,
    the code re-checks the schema before deciding that creation failed.
    """
    cache_key = (collection_name, field_name)

    if cache_key in _ready_indexes:
        return

    with _index_lock:
        if cache_key in _ready_indexes:
            return

        if not _collection_exists(collection_name):
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' does not exist."
            )

        info = client.get_collection(collection_name=collection_name)
        payload_schema = info.payload_schema or {}
        existing = payload_schema.get(field_name)

        expected_type = getattr(field_schema, "value", field_schema)
        expected_type = str(expected_type).lower()

        if existing is not None:
            actual_type = _payload_schema_type(existing)

            if actual_type and actual_type != expected_type:
                raise RuntimeError(
                    "Qdrant payload index type mismatch: "
                    f"{collection_name}.{field_name} must be "
                    f"{expected_type}, found {actual_type}."
                )

            _ready_indexes.add(cache_key)
            return

        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            # Another process may have created the index after our first check.
            info = client.get_collection(collection_name=collection_name)
            payload_schema = info.payload_schema or {}

            if field_name not in payload_schema:
                raise

        # Verify the final schema explicitly.
        info = client.get_collection(collection_name=collection_name)
        payload_schema = info.payload_schema or {}
        existing = payload_schema.get(field_name)

        if existing is None:
            raise RuntimeError(
                "Qdrant payload index creation did not complete: "
                f"{collection_name}.{field_name}"
            )

        actual_type = _payload_schema_type(existing)

        if actual_type and actual_type != expected_type:
            raise RuntimeError(
                "Qdrant payload index type mismatch after creation: "
                f"{collection_name}.{field_name} must be "
                f"{expected_type}, found {actual_type}."
            )

        _ready_indexes.add(cache_key)


# ---------------------------------------------------------------------
# Collection initialization
# ---------------------------------------------------------------------

def create_photo_embeddings_collection():
    """
    Safely create photo_embeddings if it does not exist.

    Unlike recreate_collection(), this does not delete existing vectors.
    """
    if not _collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE,
            ),
        )

    create_index()


def create_video_embeddings_collection():
    """
    Safely create video_embeddings if it does not exist.
    """
    if not _collection_exists(QDRANT_COLLECTION_VIDEO):
        client.create_collection(
            collection_name=QDRANT_COLLECTION_VIDEO,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE,
            ),
        )

    create_video_index()


def create_index():
    """
    Required payload indexes for photo_embeddings.
    """
    ensure_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="photo_id",
        field_schema=PayloadSchemaType.INTEGER,
    )
    ensure_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="event_id",
        field_schema=PayloadSchemaType.INTEGER,
    )


def create_video_index():
    """
    Required payload indexes for video_embeddings.
    """
    ensure_payload_index(
        collection_name=QDRANT_COLLECTION_VIDEO,
        field_name="video_id",
        field_schema=PayloadSchemaType.INTEGER,
    )
    ensure_payload_index(
        collection_name=QDRANT_COLLECTION_VIDEO,
        field_name="event_id",
        field_schema=PayloadSchemaType.INTEGER,
    )


def ensure_text_embeddings_indexes():
    """
    Permanent schema guard for text redundancy filtering.

    The text_embeddings collection is created elsewhere because its vector
    dimension depends on the text embedding model. This helper only guarantees
    the event_id payload index required by filtered searches.
    """
    ensure_payload_index(
        collection_name=QDRANT_COLLECTION_TEXT,
        field_name="event_id",
        field_schema=PayloadSchemaType.INTEGER,
    )


def ensure_all_existing_indexes():
    """
    Optional startup/diagnostic helper.

    It only touches collections that already exist.
    """
    if _collection_exists(QDRANT_COLLECTION):
        create_index()

    if _collection_exists(QDRANT_COLLECTION_VIDEO):
        create_video_index()

    if _collection_exists(QDRANT_COLLECTION_TEXT):
        ensure_text_embeddings_indexes()


# ---------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------

def _qdrant_rest_base() -> str:
    """
    Build the REST base URL without accidentally appending :6333 twice.
    """
    parsed = urlsplit(QDRANT_URL)

    if parsed.port is not None:
        return QDRANT_URL.rstrip("/")

    netloc = parsed.netloc
    if QDRANT_PORT:
        netloc = f"{netloc}:{QDRANT_PORT}"

    return urlunsplit(
        (
            parsed.scheme,
            netloc,
            parsed.path.rstrip("/"),
            "",
            "",
        )
    ).rstrip("/")


def qdrant_rest_call(
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    """
    Generic Qdrant REST API caller.
    """
    base = _qdrant_rest_base()
    url = base + (path if path.startswith("/") else f"/{path}")

    headers = {
        "api-key": QDRANT_API_KEY,
        "Content-Type": "application/json",
    }

    resp = requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if not resp.ok:
        try:
            err_json = resp.json()
        except Exception:
            err_json = {"raw": resp.text}

        raise RuntimeError(
            f"Qdrant REST error {resp.status_code} {resp.reason}\n"
            f"URL: {url}\n"
            f"Response: {err_json}"
        )

    if resp.text.strip():
        return resp.json()

    return {}


def qdrant_search_points(
    *,
    vector: list[float],
    limit: int = 10,
    event_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Search photo_embeddings with an optional event_id filter.
    """
    create_index()

    search_payload: Dict[str, Any] = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
    }

    if event_id is not None:
        search_payload["filter"] = {
            "must": [
                {
                    "key": "event_id",
                    "match": {"value": int(event_id)},
                }
            ]
        }

    return qdrant_rest_call(
        "POST",
        f"/collections/{QDRANT_COLLECTION}/points/search",
        payload=search_payload,
    )


def qdrant_search_video_points(
    *,
    vector: list[float],
    limit: int = 10,
    event_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Search video_embeddings with an optional event_id filter.
    """
    create_video_index()

    search_payload: Dict[str, Any] = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
    }

    if event_id is not None:
        search_payload["filter"] = {
            "must": [
                {
                    "key": "event_id",
                    "match": {"value": int(event_id)},
                }
            ]
        }

    return qdrant_rest_call(
        "POST",
        f"/collections/{QDRANT_COLLECTION_VIDEO}/points/search",
        payload=search_payload,
    )


def qdrant_search_text_points(
    *,
    vector: list[float],
    limit: int = 10,
    event_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Search text_embeddings.

    The event_id payload index is guaranteed before a filtered search.
    """
    ensure_text_embeddings_indexes()

    search_payload: Dict[str, Any] = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
    }

    if event_id is not None:
        search_payload["filter"] = {
            "must": [
                {
                    "key": "event_id",
                    "match": {"value": int(event_id)},
                }
            ]
        }

    return qdrant_rest_call(
        "POST",
        f"/collections/{QDRANT_COLLECTION_TEXT}/points/search",
        payload=search_payload,
    )


def qdrant_healthcheck() -> Dict[str, Any]:
    """
    Verify that the configured Qdrant credentials can list collections.
    """
    return qdrant_rest_call("GET", "/collections")