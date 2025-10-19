import logging
from typing import List, Dict, Any, Optional
import numpy as np
import uuid
import json

from src.embeddings.embedding_generator import EmbeddedChunk

logger = logging.getLogger(__name__)


class InMemoryVectorDB:
    """A tiny in-memory vector DB used for development and tests.

    It implements a compatible subset of the MilvusVectorDB interface used
    by the rest of the project: insert_embeddings, search, delete_collection,
    get_chunk_by_id, close.
    """

    def __init__(self, collection_name: str = "in_memory", embedding_dim: int = 384):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._store: Dict[str, Dict[str, Any]] = {}

    def insert_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> List[str]:
        ids = []
        for ec in embedded_chunks:
            data = ec.to_vector_db_format()
            vid = data['id']
            vector = np.array(data['vector'], dtype=np.float32)
            self._store[vid] = {
                'id': vid,
                'vector': vector,
                'content': data.get('content', ''),
                'citation': {
                    'source_file': data.get('source_file'),
                    'source_type': data.get('source_type'),
                    'page_number': data.get('page_number'),
                    'chunk_index': data.get('chunk_index'),
                    'start_char': data.get('start_char'),
                    'end_char': data.get('end_char')
                },
                'metadata': data.get('metadata', {}),
                'embedding_model': data.get('embedding_model', '')
            }
            ids.append(vid)
        logger.info(f"InMemoryVectorDB: inserted {len(ids)} items")
        return ids

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return -1.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return -1.0
        return float(np.dot(a, b) / denom)

    def search(self, query_vector: List[float], limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        qv = np.array(query_vector, dtype=np.float32)
        results = []
        for vid, entry in self._store.items():
            score = self._cosine_sim(qv, entry['vector'])
            results.append({
                'id': vid,
                'score': score,
                'content': entry['content'],
                'citation': entry['citation'],
                'metadata': entry.get('metadata', {}),
                'embedding_model': entry.get('embedding_model', '')
            })

        # sort by descending similarity
        results.sort(key=lambda r: r['score'], reverse=True)
        return results[:limit]

    def delete_collection(self):
        self._store.clear()
        logger.info("InMemoryVectorDB: cleared collection")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        entry = self._store.get(chunk_id)
        if not entry:
            return None
        return {
            'id': entry['id'],
            'content': entry['content'],
            'metadata': entry.get('metadata', {}),
            'source_file': entry['citation'].get('source_file'),
            'source_type': entry['citation'].get('source_type'),
            'page_number': entry['citation'].get('page_number'),
            'chunk_index': entry['citation'].get('chunk_index')
        }

    def list_items(self) -> List[Dict[str, Any]]:
        """Return a serializable list of all stored items.

        Each item contains the id, content, citation fields, metadata and
        embedding model name. This is intended for UI listing/exporting.
        """
        items: List[Dict[str, Any]] = []
        for vid, entry in self._store.items():
            items.append({
                'id': entry['id'],
                'content': entry['content'],
                'source_file': entry['citation'].get('source_file'),
                'source_type': entry['citation'].get('source_type'),
                'page_number': entry['citation'].get('page_number'),
                'chunk_index': entry['citation'].get('chunk_index'),
                'start_char': entry['citation'].get('start_char'),
                'end_char': entry['citation'].get('end_char'),
                'metadata': entry.get('metadata', {}),
                'embedding_model': entry.get('embedding_model', '')
            })
        return items

    def export_state(self) -> List[Dict[str, Any]]:
        """Return a full serializable snapshot of the store including vectors.

        The vector is converted to a plain list for JSON serialization.
        """
        export: List[Dict[str, Any]] = []
        for vid, entry in self._store.items():
            export.append({
                'id': entry['id'],
                'vector': entry['vector'].tolist() if entry.get('vector') is not None else None,
                'content': entry.get('content', ''),
                'citation': entry.get('citation', {}),
                'metadata': entry.get('metadata', {}),
                'embedding_model': entry.get('embedding_model', '')
            })
        return export

    def import_state(self, items: List[Dict[str, Any]]) -> int:
        """Import a snapshot previously created by export_state.

        Returns the number of items imported.
        """
        imported = 0
        for it in items:
            # Use provided id when available. If missing, generate a deterministic
            # UUID5 based on the content, citation and embedding_model. This
            # prevents creating a new random id on every import which would
            # otherwise break UI elements that reference chunk ids.
            provided_id = it.get('id')
            if provided_id:
                vid = provided_id
            else:
                try:
                    name_seed = (
                        (it.get('content') or '') + '|' +
                        json.dumps(it.get('citation') or {}, sort_keys=True) + '|' +
                        str(it.get('embedding_model') or '')
                    )
                    vid = str(uuid.uuid5(uuid.NAMESPACE_URL, name_seed))
                    logger.debug(f"InMemoryVectorDB: generated deterministic id {vid} for imported item")
                except Exception:
                    # Fallback to random uuid if deterministic generation fails
                    vid = str(uuid.uuid4())
            vec = it.get('vector')
            if vec is None:
                # skip items without vector
                continue
            try:
                vector = np.array(vec, dtype=np.float32)
            except Exception:
                continue

            self._store[vid] = {
                'id': vid,
                'vector': vector,
                'content': it.get('content', ''),
                'citation': it.get('citation', {}),
                'metadata': it.get('metadata', {}),
                'embedding_model': it.get('embedding_model', '')
            }
            imported += 1

        logger.info(f"InMemoryVectorDB: imported {imported} items")
        return imported

    def close(self):
        # nothing to close in memory
        logger.info("InMemoryVectorDB: closed")
