"""MongoDB-backed pHash fraud DB bootstrap utility.

Provides a minimal schema and convenience functions to store and query
fraudulent image pHashes. This is STEP 1 (bootstrap stage) for pHash
production integration: exact-match storage and lookup only.

Functions:
- connect_db(uri=None, db_name='phash_db', collection_name='fraud_phashes') -> collection
    Connects to MongoDB and ensures a unique index on `phash_hex`.

- insert_phash(phash_hex, image_id, source, uri=None)
    Inserts a fraud record (image_id, phash_hex, source, label='fraud', created_at).
    Returns the inserted document id or None on duplicate.

- get_all_phashes(uri=None)
    Returns a list of stored fraud records (dictionaries).

- phash_exists(phash_hex, uri=None) -> bool
    Checks whether an exact `phash_hex` match exists in the DB.

Note: this module intentionally implements only exact-match storage and
lookup to keep the bootstrap simple and auditable. Similarity/ANN search
will be implemented later in the next step.
"""
from datetime import datetime
from typing import Optional, List, Dict
import logging

try:
    from pymongo import MongoClient, errors
except Exception as e:
    MongoClient = None
    errors = None
    logging.warning('pymongo not available: %s', e)


DEFAULT_URI = 'mongodb://localhost:27017'


def connect_db(uri: Optional[str] = None, db_name: str = 'phash_db', collection_name: str = 'fraud_phashes'):
    """Connect to MongoDB and return the collection object.

    Ensures that `phash_hex` has a unique index. If pymongo is not
    installed or the server is not reachable, raises a RuntimeError with a
    human-readable message.
    """
    if MongoClient is None:
        raise RuntimeError('pymongo is not installed; please `pip install pymongo`')
    uri = uri or DEFAULT_URI
    client = MongoClient(uri)
    try:
        db = client[db_name]
        coll = db[collection_name]
        # Create a unique index on phash_hex to prevent duplicates
        coll.create_index('phash_hex', unique=True)
        return coll
    except Exception as e:
        client.close()
        raise RuntimeError(f'Failed to connect to MongoDB at {uri}: {e}')


def insert_phash(phash_hex: str, image_id: str, source: str, uri: Optional[str] = None) -> Optional[str]:
    """Insert a fraud pHash record into the DB.

    Returns the inserted document's `_id` (as str) on success, or None if the
    phash already exists (duplicate). Raises RuntimeError on connection errors.
    """
    coll = connect_db(uri)
    doc = {
        'image_id': str(image_id),
        'phash_hex': str(phash_hex),
        'source': str(source),
        'label': 'fraud',
        'created_at': datetime.utcnow(),
    }
    try:
        res = coll.insert_one(doc)
        logging.info('Inserted phash %s for image %s', phash_hex, image_id)
        return str(res.inserted_id)
    except errors.DuplicateKeyError:
        logging.info('phash %s already exists in DB', phash_hex)
        return None
    except Exception as e:
        raise RuntimeError(f'Failed to insert phash into DB: {e}')


def get_all_phashes(uri: Optional[str] = None) -> List[Dict]:
    """Return all fraud pHash records as a list of dictionaries.

    Raises RuntimeError on connection errors.
    """
    coll = connect_db(uri)
    docs = list(coll.find({}, {'_id': 0, 'image_id': 1, 'phash_hex': 1, 'source': 1, 'label': 1, 'created_at': 1}))
    return docs


def phash_exists(phash_hex: str, uri: Optional[str] = None) -> bool:
    """Check if an exact phash_hex exists in the DB (True/False).

    This is an exact-match lookup and intentionally fast. For similarity
    queries we'll add ANN search later.
    """
    coll = connect_db(uri)
    return coll.count_documents({'phash_hex': str(phash_hex)}, limit=1) > 0


# -------------------------------
# Similarity helpers (STEP 2)
# -------------------------------

def _normalize_hex(h: str) -> str:
    """Normalize hex string: remove 0x, lower, pad to even length."""
    if not h:
        return ''
    s = str(h).lower().strip()
    if s.startswith('0x'):
        s = s[2:]
    # ensure even length for bytes.fromhex
    if len(s) % 2 == 1:
        s = '0' + s
    return s


def hamming_distance_hex(a: str, b: str) -> int:
    """Compute Hamming distance (number of differing bits) between two hex strings.

    Steps:
    - Normalize hex strings, convert to bytes, XOR, and count set bits.
    - If lengths differ, pad the shorter one with leading zeros (implicit in normalization).
    """
    a_n = _normalize_hex(a)
    b_n = _normalize_hex(b)
    try:
        a_bytes = bytes.fromhex(a_n)
        b_bytes = bytes.fromhex(b_n)
    except ValueError as e:
        raise ValueError(f'Invalid hex string: {e}')

    # Pad the shorter bytes with leading zeros
    if len(a_bytes) < len(b_bytes):
        a_bytes = b'\x00' * (len(b_bytes) - len(a_bytes)) + a_bytes
    elif len(b_bytes) < len(a_bytes):
        b_bytes = b'\x00' * (len(a_bytes) - len(b_bytes)) + b_bytes

    dist = 0
    for x, y in zip(a_bytes, b_bytes):
        dist += (x ^ y).bit_count()
    return dist


def find_similar_phash(query_phash_hex: str, max_distance_exact: int = 5, max_distance_near: int = 10, uri: Optional[str] = None) -> Dict:
    """Search stored fraud pHashes for similarity to query.

    - Iterates over stored fraud pHashes (simple, CPU-only linear scan).
    - Computes Hamming distance to each stored phash.
    - Returns a structured dict with fields:
      { match: bool, match_type: 'exact'|'near'|'none', min_distance: int|None, matched_image_id: str|None }

    Matching rules:
    - distance <= max_distance_exact  => 'exact'
    - distance <= max_distance_near   => 'near'
    - else                            => 'none'

    Note: this is intentionally straightforward and explainable. For large
    databases, we'll move to ANN-based search (FAISS/HNSW) in a later step.
    """
    docs = get_all_phashes(uri)
    if not docs:
        return {'match': False, 'match_type': 'none', 'min_distance': None, 'matched_image_id': None}

    min_distance = None
    matched_image_id = None
    matched_source = None

    for d in docs:
        stored = d.get('phash_hex')
        if not stored:
            continue
        try:
            dist = hamming_distance_hex(query_phash_hex, stored)
        except ValueError:
            # skip invalid stored values
            continue
        if min_distance is None or dist < min_distance:
            min_distance = dist
            matched_image_id = d.get('image_id')
            matched_source = d.get('source')

    if min_distance is None:
        return {'match': False, 'match_type': 'none', 'min_distance': None, 'matched_image_id': None}

    if min_distance <= max_distance_exact:
        match_type = 'exact'
        match = True
    elif min_distance <= max_distance_near:
        match_type = 'near'
        match = True
    else:
        match_type = 'none'
        match = False

    return {'match': match, 'match_type': match_type, 'min_distance': int(min_distance), 'matched_image_id': matched_image_id}


if __name__ == '__main__':
    # Demo: insert 2 dummy fraud hashes, then query identical, near, and unrelated hashes.
    print('pHash DB similarity demo: attempting to connect to MongoDB...')
    try:
        demo_hashes = [
            ('deadbeefcafebabe', 'IMG_DEMO_001', 'CASIA2'),
            ('0123456789abcdef', 'IMG_DEMO_002', 'CASIA2'),
        ]
        for phash_hex, image_id, source in demo_hashes:
            inserted = insert_phash(phash_hex, image_id, source)
            if inserted:
                print(f'Inserted: {phash_hex} (image_id={image_id})')
            else:
                print(f'Exists already: {phash_hex}')

        queries = {
            'identical': 'deadbeefcafebabe',
            'slightly_modified': 'deadbeefcafebabf',  # one-bit-ish change
            'unrelated': 'ffffffffffffffff',
        }

        print('\nSimilarity queries:')
        for name, q in queries.items():
            res = find_similar_phash(q)
            print(f' Query [{name}] ({q}) -> match={res["match"]}, type={res["match_type"]}, min_distance={res["min_distance"]}, matched_image_id={res["matched_image_id"]}')

    except RuntimeError as e:
        print('Demo aborted:', e)
        print('If you want to run the demo, ensure MongoDB is reachable at mongodb://localhost:27017 or pass a --uri to connect_db.')
