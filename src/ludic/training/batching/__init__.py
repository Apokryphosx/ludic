from .rollout_engine import RolloutEngine
from .synced_batching import RolloutBatchSource
try:
    from .pipeline import PipelineBatchSource, run_pipeline_actor
except ImportError:
    # Redis is optional; import will fail if redis is not installed.
    PipelineBatchSource = None  # type: ignore
    run_pipeline_actor = None  # type: ignore
from .intra_batch_control import (
    RequestStrategy, 
    IdentityStrategy, 
    GRPORequestStrategy
)
from .requests_from_dataset import (
    RequestsExhausted,
    make_requests_fn_from_queue,
    make_dataset_queue_requests_fn,
    make_dataset_sequence_requests_fn,
)

__all__ = [
    "RolloutEngine",
    "RolloutBatchSource",
    "PipelineBatchSource",
    "run_pipeline_actor",
    "RequestStrategy",
    "IdentityStrategy",
    "GRPORequestStrategy",
    "RequestsExhausted",
    "make_requests_fn_from_queue",
    "make_dataset_queue_requests_fn",
    "make_dataset_sequence_requests_fn",
]
