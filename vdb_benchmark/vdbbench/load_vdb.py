#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import time
import numpy as np
from typing import Optional, Tuple

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Add the parent directory to sys.path to import config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdbbench.config_loader import load_config, merge_config_with_args
from vdbbench.compact_and_watch import monitor_progress

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


###############################################################################
# GT (Ground Truth) helper / reasoning
###############################################################################
"""
Why a separate GT collection?
- To measure recall@K across approximate indexes (DiskANN/HNSW/AISAQ), you need an
  "exact" top-K reference. In Milvus, that typically means a FLAT index (exact).
- GT should be per DATASET (vectors+ids+metric), not per index type. If the dataset
  is identical, the GT should be shared across DiskANN/HNSW/AISAQ comparisons.

What safeguard do we implement?
- Deterministic GT naming from dataset signature (default: N, dim, distribution, seed, metric).
- Reuse GT if it already exists.
- CRITICAL: If GT exists and is already populated with the expected entity count,
  we skip inserting into GT again. This prevents the GT from accidentally growing
  to 2x/3x/…N across repeated runs.

We also validate schema on GT reuse:
- id must be primary, auto_id=False
- vector dtype must be FLOAT_VECTOR
- dim must match

If schema/count mismatch is detected, we fail fast unless --force-gt is used.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Load vectors into Milvus database (optionally create GT FLAT)")

    # Connection parameters
    parser.add_argument("--host", type=str, default="localhost", help="Milvus server host")
    parser.add_argument("--port", type=str, default="19530", help="Milvus server port")

    # Collection parameters
    parser.add_argument("--collection-name", type=str, required=False, help="Name of the collection to create")
    parser.add_argument("--dimension", type=int, required=False, help="Vector dimension")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards for the collection")
    parser.add_argument(
        "--vector-dtype",
        type=str,
        default="float",
        choices=["float"],
        help="Vector data type (currently only FLOAT_VECTOR supported in this loader)",
    )

    # Force flags
    parser.add_argument("--force", action="store_true", help="Force recreate MAIN collection if it exists")
    parser.add_argument("--force-gt", action="store_true", help="Force recreate GT collection if it exists")

    # Data generation parameters
    parser.add_argument("--num-vectors", type=int, required=False, help="Number of vectors to generate")
    parser.add_argument("--distribution", type=str, default="uniform", choices=["uniform", "normal"],
                        help="Distribution for vector generation")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for reproducible vector generation")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for insertion")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Vectors to generate per chunk")

    # Index parameters
    parser.add_argument("--index-type", type=str, default="DISKANN", help="Index type")
    parser.add_argument("--metric-type", type=str, default="COSINE", help="Metric type for index")
    parser.add_argument("--max-degree", type=int, default=16, help="DiskANN MaxDegree parameter")
    parser.add_argument("--search-list-size", type=int, default=200, help="DiskANN SearchListSize parameter")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction parameter")
    parser.add_argument("--inline-pq", type=int, default=16, help="AISAQ inline_pq parameter")

    # GT options
    parser.add_argument("--create-gt", action="store_true",
                        help="Also create/populate a GT collection with FLAT index (exact)")
    parser.add_argument("--gt-collection-name", type=str, default=None,
                        help="Optional override GT collection name. If omitted, auto-generated deterministically.")
    parser.add_argument("--gt-num-shards", type=int, default=None,
                        help="GT number of shards (default: same as --num-shards)")
    parser.add_argument("--gt-metric-type", type=str, default=None,
                        help="GT metric type (default: same as --metric-type)")
    parser.add_argument("--gt-key-mode", type=str, default="signature", choices=["signature", "nd"],
                        help="How to build auto GT name: 'signature' (safe) or 'nd' (coarse)")

    # Monitoring parameters
    parser.add_argument("--monitor-interval", type=int, default=5, help="Seconds between monitoring checks")
    parser.add_argument("--compact", action="store_true", help="Perform compaction after loading")

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # What-if option to print args and exit
    parser.add_argument("--what-if", action="store_true", help="Print the arguments after processing and exit")

    # Debug option
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)

    if args.what_if:
        logger.info("What-if mode: printing args and exiting.")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        sys.exit(0)

    required_params = ["collection_name", "dimension", "num_vectors"]
    missing = [p for p in required_params if getattr(args, p, None) is None]
    if missing:
        parser.error(f"Missing required parameters: {', '.join(missing)}")

    return args


def connect_to_milvus(host, port) -> bool:
    try:
        connections.connect(
            "default",
            host=host,
            port=port,
            max_receive_message_length=514_983_574,
            max_send_message_length=514_983_574,
        )
        logger.info(f"Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Error connecting to Milvus: {e}")
        return False


def create_collection(collection_name: str, dim: int, num_shards: int, vector_dtype, force: bool = False) -> Optional[Collection]:
    try:
        if utility.has_collection(collection_name):
            if force:
                Collection(name=collection_name).drop()
                logger.info(f"Dropped existing collection: {collection_name}")
            else:
                logger.warning(f"Collection '{collection_name}' already exists. (no force) Reusing it.")
                return Collection(name=collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=vector_dtype, dim=dim),
        ]
        schema = CollectionSchema(fields, description="Benchmark Collection")
        col = Collection(name=collection_name, schema=schema, num_shards=num_shards)
        logger.info(f"Created collection '{collection_name}' (dim={dim}, shards={num_shards})")
        return col
    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {e}")
        return None


def create_index(collection: Collection, index_params: dict) -> bool:
    try:
        logger.info(f"Creating index on '{collection.name}' with params: {index_params}")
        collection.create_index("vector", index_params)
        return True
    except Exception as e:
        logger.error(f"Failed to create index on '{collection.name}': {e}")
        return False


def flush_collection(collection: Collection) -> None:
    # Flush the collection
    t0 = time.time()
    collection.flush()
    logger.info(f"Flush '{collection.name}' completed in {time.time() - t0:.2f}s")


def generate_vectors(num_vectors: int, dim: int, distribution: str = "uniform", normalize: bool = False) -> list:
    """Generate random vectors.
    
    Args:
        num_vectors: Number of vectors to generate
        dim: Vector dimension
        distribution: 'uniform' or 'normal' or 'zipfian'
        normalize: If True, normalize vectors (recommended for COSINE metric)
    """
    if distribution == "uniform":
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
    elif distribution == "normal":
        vectors = np.random.normal(0, 1, (num_vectors, dim)).astype(np.float32)
    elif distribution == 'zipfian':
        # Simplified zipfian-like distribution
        base = np.random.random((num_vectors, dim)).astype(np.float32)
        skew = np.random.zipf(1.5, (num_vectors, 1)).astype(np.float32)
        vectors = base * (skew / 10)
    else:
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)

    # Normalize only if requested (typically for COSINE metric)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        vectors = vectors / norms
        
    return vectors.tolist()


def insert_data(
    main_col: Collection,
    vectors: list,
    batch_size: int,
    id_offset: int,
    gt_col: Optional[Collection],
    gt_should_insert: bool,
    max_retries: int = 3,
) -> int:
    """Insert vectors into collections with retry logic.
    
    Args:
        main_col: Main collection to insert into
        vectors: List of vectors to insert
        batch_size: Size of each batch
        id_offset: Starting ID offset
        gt_col: Optional GT collection
        gt_should_insert: Whether to insert into GT
        max_retries: Maximum retry attempts per batch
        
    Returns:
        Number of vectors successfully inserted
    """
    total = len(vectors)
    num_batches = (total + batch_size - 1) // batch_size
    inserted = 0
    failed_batches = 0
    t0 = time.time()

    for i in range(num_batches):
        bs = i * batch_size
        be = min((i + 1) * batch_size, total)
        ids = list(range(id_offset + bs, id_offset + be))
        batch_vectors = vectors[bs:be]

        # Retry logic for main collection
        success = False
        for attempt in range(max_retries):
            try:
                main_col.insert([ids, batch_vectors])
                success = True
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Batch {i+1} insert failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Batch {i+1} insert failed after {max_retries} attempts: {e}")
                    failed_batches += 1
        
        if not success:
            continue
            
        # Insert into GT if needed (with retry)
        if gt_col is not None and gt_should_insert:
            for attempt in range(max_retries):
                try:
                    gt_col.insert([ids, batch_vectors])
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"GT batch {i+1} insert failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(f"GT batch {i+1} insert failed after {max_retries} attempts: {e}")

        inserted += (be - bs)

        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            elapsed = time.time() - t0
            rate = inserted / elapsed if elapsed > 0 else 0
            logger.info(f"Inserted {inserted:,}/{total:,} (batch {i+1}/{num_batches}), rate={rate:.1f} vec/s")
    
    if failed_batches > 0:
        logger.error(f"WARNING: {failed_batches} batches failed to insert!")

    return inserted


def build_gt_name(args) -> str:
    metric = (args.gt_metric_type or args.metric_type)
    metric = str(metric).upper()
    if args.gt_key_mode == "nd":
        return f"gt_n{args.num_vectors}_dim{args.dimension}_{metric}_flat"
    return f"gt_n{args.num_vectors}_dim{args.dimension}_{args.distribution}_seed{args.seed}_{metric}_flat"


def _validate_gt_schema_or_die(gt: Collection, expected_dim: int) -> None:
    """
    Fail fast if GT schema is not compatible with this loader’s expected format.
    This catches the common case where an old GT was created with auto_id=True,
    or the vector field is BINARY, etc.
    """
    try:
        fields = {f.name: f for f in gt.schema.fields}
        if "id" not in fields or "vector" not in fields:
            raise SystemExit(f"GT '{gt.name}' schema mismatch: expected fields 'id' and 'vector'.")

        idf = fields["id"]
        vf = fields["vector"]

        if not getattr(idf, "is_primary", False):
            raise SystemExit(f"GT '{gt.name}' schema mismatch: 'id' is not primary.")
        if getattr(idf, "auto_id", None) is True:
            raise SystemExit(f"GT '{gt.name}' schema mismatch: 'id' has auto_id=True; must be auto_id=False.")

        if vf.dtype != DataType.FLOAT_VECTOR:
            raise SystemExit(f"GT '{gt.name}' schema mismatch: vector dtype is {vf.dtype}, expected FLOAT_VECTOR.")
        dim = vf.params.get("dim", None)
        if int(dim) != int(expected_dim):
            raise SystemExit(f"GT '{gt.name}' schema mismatch: dim={dim}, expected {expected_dim}.")
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Failed validating GT schema for '{gt.name}': {e}")


def ensure_gt_collection(args, vector_dtype) -> Tuple[Optional[Collection], Optional[str], bool]:
    """
    Create or reuse GT collection.
    Returns (gt_collection, gt_name, gt_should_insert).

    gt_should_insert means: during this run, should we insert vectors into GT?
    """
    if not args.create_gt:
        return None, None, False

    gt_name = args.gt_collection_name or build_gt_name(args)
    args.gt_collection_name = gt_name

    gt_shards = args.gt_num_shards if args.gt_num_shards is not None else args.num_shards
    gt_metric = args.gt_metric_type if args.gt_metric_type is not None else args.metric_type
    
    # Validate GT metric matches main collection metric
    if str(gt_metric).upper() != str(args.metric_type).upper():
        logger.warning(
            f"⚠️  GT metric '{gt_metric}' differs from main collection metric '{args.metric_type}'. "
            f"This will produce invalid recall measurements! GT should use same metric."
        )

    # If exists and not force-gt: reuse with safeguards
    if utility.has_collection(gt_name) and not args.force_gt:
        gt = Collection(gt_name)
        gt.load()

        # Validate schema before reusing (prevents accidental "append forever" GTs)
        _validate_gt_schema_or_die(gt, expected_dim=args.dimension)

        existing = gt.num_entities
        expected = int(args.num_vectors)

        if existing == expected:
            logger.info(f"GT '{gt_name}' already populated ({existing:,} entities). Reusing; skipping GT insertion.")
            return gt, gt_name, False
        if existing == 0:
            logger.info(f"GT '{gt_name}' exists but empty. Will populate it now.")
            return gt, gt_name, True

        # Anything else is suspicious / poisonous
        raise SystemExit(
            f"GT '{gt_name}' exists with {existing:,} entities, expected {expected:,}. "
            f"Refusing to append/mix datasets. Use --force-gt to rebuild GT."
        )

    # Create/recreate GT
    gt = create_collection(gt_name, args.dimension, gt_shards, vector_dtype, force=True if args.force_gt else False)
    if gt is None:
        return None, None, False

    gt_index = {"index_type": "FLAT", "metric_type": str(gt_metric).upper(), "params": {}}
    if not create_index(gt, gt_index):
        return None, None, False

    # New GT needs population
    return gt, gt_name, True


def main():
    args = parse_args()

    if not connect_to_milvus(args.host, args.port):
        return 1

    # Deterministic generation
    np.random.seed(args.seed)

    vector_dtype = DataType.FLOAT_VECTOR

    # Create / reuse main collection
    main_col = create_collection(args.collection_name, args.dimension, args.num_shards, vector_dtype, force=args.force)
    if main_col is None:
        return 1

    # Create main index
    index_params = {"index_type": args.index_type, "metric_type": str(args.metric_type).upper(), "params": {}}
    if args.index_type == "HNSW":
        index_params["params"] = {"M": args.M, "efConstruction": args.ef_construction}
    elif args.index_type == "DISKANN":
        index_params["params"] = {"max_degree": args.max_degree, "search_list_size": args.search_list_size}
    elif args.index_type == "AISAQ":
        index_params["params"] = {"inline_pq": args.inline_pq, "max_degree": args.max_degree,
                                  "search_list_size": args.search_list_size}
    else:
        logger.error(f"Unsupported index_type: {args.index_type}")
        return 1

    if not create_index(main_col, index_params):
        return 1

    # Create/reuse GT (FLAT) if requested
    gt_col, gt_name, gt_should_insert = ensure_gt_collection(args, vector_dtype)
    if args.create_gt and gt_col is None:
        logger.error("Requested --create-gt but failed to create/reuse GT collection.")
        return 1

    if gt_col is not None:
        logger.info(f"Using GT collection: {gt_name} (gt_should_insert={gt_should_insert})")

    # Determine if we should normalize vectors based on metric type
    normalize_vectors = str(args.metric_type).upper() == "COSINE"
    if normalize_vectors:
        logger.info("Vector normalization enabled (COSINE metric)")
    else:
        logger.info(f"Vector normalization disabled ({args.metric_type} metric)")

    # Generate + insert in chunks
    total_vectors = args.num_vectors
    remaining = total_vectors
    global_offset = 0
    chunk_idx = 0

    logger.info(
        f"Generating+inserting {total_vectors:,} vectors (dim={args.dimension}, dist={args.distribution}, seed={args.seed})"
    )

    t_all = time.time()
    while remaining > 0:
        chunk_idx += 1
        chunk_size = min(args.chunk_size, remaining)
        logger.info(f"Chunk {chunk_idx}: generating {chunk_size:,} vectors")
        t0 = time.time()
        vecs = generate_vectors(chunk_size, args.dimension, args.distribution, normalize=normalize_vectors)
        logger.info(f"Chunk {chunk_idx}: generated in {time.time() - t0:.2f}s")

        logger.info(
            f"Chunk {chunk_idx}: inserting into '{args.collection_name}'"
            + (f" and GT '{gt_name}'" if (gt_col is not None and gt_should_insert) else "")
            + f" (id_offset={global_offset})"
        )
        inserted = insert_data(main_col, vecs, args.batch_size, global_offset, gt_col, gt_should_insert)
        global_offset += inserted
        remaining -= chunk_size

    logger.info(f"All chunks inserted in {time.time() - t_all:.2f}s")

    flush_collection(main_col)
    if gt_col is not None and gt_should_insert:
        flush_collection(gt_col)
    
    # Verify collections are loaded and ready
    logger.info(f"Loading collection '{args.collection_name}' into memory...")
    try:
        main_col.load()
        logger.info(f"Collection '{args.collection_name}' loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load collection '{args.collection_name}': {e}")
        return 1
    
    if gt_col is not None:
        logger.info(f"Loading GT collection '{gt_name}' into memory...")
        try:
            gt_col.load()
            logger.info(f"GT collection '{gt_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GT collection '{gt_name}': {e}")
            return 1

    # Monitor index build progress
    logger.info(f"Monitoring index build for '{args.collection_name}' every {args.monitor_interval}s (this may take a while)")
    monitor_progress(args.collection_name, args.monitor_interval, zero_threshold=10)
    
    # Verify final entity count
    final_count = main_col.num_entities
    if final_count != args.num_vectors:
        logger.warning(f"⚠️  Expected {args.num_vectors:,} entities, but collection has {final_count:,}")

    if gt_col is not None:
        logger.info(f"Monitoring index build for GT '{gt_name}' every {args.monitor_interval}s")
        monitor_progress(gt_name, args.monitor_interval, zero_threshold=10)

    # Optional compaction
    if args.compact:
        logger.info(f"Compacting '{args.collection_name}'")
        main_col.compact()
        monitor_progress(args.collection_name, args.monitor_interval, zero_threshold=30)

        if gt_col is not None and gt_should_insert:
            logger.info(f"Compacting GT '{gt_name}'")
            gt_col.compact()
            monitor_progress(gt_name, args.monitor_interval, zero_threshold=30)

    logger.info("Load completed successfully.")
    if gt_col is not None:
        logger.info(f"GT ready: {gt_name} (reused={not gt_should_insert})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
