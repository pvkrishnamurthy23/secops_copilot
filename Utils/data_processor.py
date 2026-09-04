# pipeline/data_processor.py

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Generator, Optional

logger = logging.getLogger(__name__)


# ─── Enums & Constants ────────────────────────────────────────────────────────

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class TransformType(Enum):
    FILTER = "filter"
    MAP = "map"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"


BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


# ─── Decorators ───────────────────────────────────────────────────────────────

def retry(max_attempts: int = MAX_RETRIES, backoff: float = RETRY_BACKOFF_BASE):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    wait = backoff ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
            raise RuntimeError(f"All {max_attempts} attempts failed.") from last_error
        return wrapper
    return decorator


def timed(func: Callable):
    """Log execution time of async functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Record:
    id: str
    source: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    processed_at: Optional[float] = None

    def mark_processed(self):
        self.processed_at = time.time()

    def add_error(self, msg: str):
        self.errors.append(msg)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


@dataclass
class PipelineResult:
    pipeline_id: str
    status: PipelineStatus
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.succeeded / self.total * 100) if self.total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": f"{self.success_rate:.1f}%",
            "duration_seconds": round(self.duration_seconds, 3),
        }


# ─── Transform Steps ──────────────────────────────────────────────────────────

class TransformStep:
    def __init__(self, name: str, transform_type: TransformType):
        self.name = name
        self.transform_type = transform_type

    async def apply(self, record: Record) -> Optional[Record]:
        raise NotImplementedError


class FilterStep(TransformStep):
    def __init__(self, name: str, predicate: Callable[[Record], bool]):
        super().__init__(name, TransformType.FILTER)
        self.predicate = predicate

    async def apply(self, record: Record) -> Optional[Record]:
        if self.predicate(record):
            return record
        logger.debug(f"Record {record.id} filtered out by {self.name}")
        return None


class MapStep(TransformStep):
    def __init__(self, name: str, transform_fn: Callable[[dict], dict]):
        super().__init__(name, TransformType.MAP)
        self.transform_fn = transform_fn

    async def apply(self, record: Record) -> Optional[Record]:
        try:
            record.payload = self.transform_fn(record.payload)
            return record
        except Exception as e:
            record.add_error(f"MapStep '{self.name}' failed: {e}")
            return record


class EnrichStep(TransformStep):
    def __init__(self, name: str, enrich_fn: Callable[[Record], dict]):
        super().__init__(name, TransformType.ENRICH)
        self.enrich_fn = enrich_fn

    async def apply(self, record: Record) -> Optional[Record]:
        try:
            extra = self.enrich_fn(record)
            record.metadata.update(extra)
            return record
        except Exception as e:
            record.add_error(f"EnrichStep '{self.name}' failed: {e}")
            return record


# ─── Pipeline Engine ──────────────────────────────────────────────────────────

class DataPipeline:
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.steps: list[TransformStep] = []
        self._stats: dict[str, int] = defaultdict(int)

    def add_step(self, step: TransformStep) -> "DataPipeline":
        """Chain steps fluently."""
        self.steps.append(step)
        return self

    def _batched(self, records: list[Record]) -> Generator[list[Record], None, None]:
        """Yield records in fixed-size batches."""
        for i in range(0, len(records), BATCH_SIZE):
            yield records[i:i + BATCH_SIZE]

    async def _process_record(self, record: Record) -> Optional[Record]:
        """Run a single record through all pipeline steps."""
        for step in self.steps:
            if record is None:
                break
            record = await step.apply(record)
        if record:
            record.mark_processed()
        return record

    async def _process_batch(self, batch: list[Record]) -> list[Optional[Record]]:
        """Process a batch of records concurrently."""
        tasks = [self._process_record(r) for r in batch]
        return await asyncio.gather(*tasks, return_exceptions=False)

    @timed
    @retry(max_attempts=MAX_RETRIES)
    async def run(self, records: list[Record]) -> PipelineResult:
        """Execute the pipeline over a list of records."""
        start = time.perf_counter()
        result = PipelineResult(
            pipeline_id=self.pipeline_id,
            status=PipelineStatus.RUNNING,
            total=len(records)
        )

        logger.info(f"Pipeline '{self.pipeline_id}' starting with {len(records)} records.")

        all_outputs: list[Optional[Record]] = []

        for batch in self._batched(records):
            outputs = await self._process_batch(batch)
            all_outputs.extend(outputs)

        for output in all_outputs:
            if output is None:
                result.skipped += 1
            elif output.is_valid:
                result.succeeded += 1
            else:
                result.failed += 1
                result.errors.extend(output.errors)

        result.duration_seconds = time.perf_counter() - start
        result.status = (
            PipelineStatus.SUCCESS if result.failed == 0
            else PipelineStatus.PARTIAL if result.succeeded > 0
            else PipelineStatus.FAILED
        )

        logger.info(f"Pipeline '{self.pipeline_id}' finished: {json.dumps(result.summary())}")
        return result


# ─── Example Usage ────────────────────────────────────────────────────────────

async def example():
    records = [
        Record(id=f"rec_{i}", source="csv_import", payload={"value": i, "label": f"item_{i}"})
        for i in range(250)
    ]

    pipeline = (
        DataPipeline("etl_demo")
        .add_step(FilterStep("drop_negatives", lambda r: r.payload.get("value", 0) >= 0))
        .add_step(MapStep("normalize_label", lambda p: {**p, "label": p["label"].upper()}))
        .add_step(EnrichStep("add_timestamp", lambda r: {"ingested_at": time.time()}))
    )

    result = await pipeline.run(records)
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example())
