import requests
import torch
from typing import List
from ludic.distributed.interfaces import ControlPlane, WeightMetadata, TensorCommunicator
from ludic.inference.vllm_client import VLLMChatClient

class VllmControlPlane(ControlPlane):
    def __init__(self, client: VLLMChatClient):
        self.client = client
        self.session = client._session
        self.url = client.server_url

    def announce_update_batch(self, metadata: List[WeightMetadata]) -> None:
        """
        Hits the new /update_param_batch endpoint.
        """
        # The server endpoint returns immediately after scheduling the task.
        # But we need to make sure the server IS listening before we start broadcasting.
        # Ideally, we would wait for an ACK, or reliance on the fact that 
        # local HTTP is faster than the time it takes to setup the NCCL call.
        
        resp = self.session.post(
            f"{self.url}/update_param_batch",
            json={"metadata": metadata},
            timeout=30.0
        )
        resp.raise_for_status()

    def finalize_update(self, version: str | None = None) -> None:
        # In the batched version, the server handles cache reset automatically 
        # at the end of the RPC call. We might poll for version change here if strictly sync.
        pass

class VllmTensorCommunicator(TensorCommunicator):
    def __init__(self, client: VLLMChatClient):
        if not client._pynccl_comm:
            raise RuntimeError("vLLM Client has no active NCCL communicator")
        self._comm = client._pynccl_comm
        self._rank = client._rank

    @property
    def rank(self) -> int:
        return self._rank

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self._comm.broadcast(tensor, src=src)

    def barrier(self) -> None:
        self._comm.group.barrier()