"""Centralized batched inference for self-play worker pools.

Measured on this M3 Pro (2026-08-18): one MPS forward costs ~1.7 ms flat
from batch 16 through 256 -- the GPU serves every worker's leaf batch
simultaneously for the price of one. Workers keep their small
quality-preserving ``leaf_batch``; only the tensor math is centralized.

Wiring: the parent spawns ``serve`` as a Process with one shared request
queue and one response queue per worker; each worker's ``MCTSAgent``
gets a :class:`RemoteEvaluator` and never touches torch. The evaluator
seam on the agent takes pre-encoded numpy arrays, so the protocol is
device- and framework-agnostic (the same server runs CUDA on a cluster
node by changing the device string).
"""

from __future__ import annotations

import queue as queue_mod
from pathlib import Path

import numpy as np

REQUEST_TIMEOUT_S = 0.05  # server poll granularity for shutdown checks


def serve(
    checkpoint: str,
    device: str,
    request_q,
    response_qs: dict[int, object],
    stop_event,
    max_batch: int = 256,
) -> None:
    """Process target: batch requests from every worker into one forward.

    A request is ``(worker_id, req_id, hex, glob, faction, cand, mask)``
    with numpy payloads; the response is ``(req_id, priors, values)``
    where ``priors`` is padded to the request's own candidate width.
    """
    import torch

    from bgai.training.model import PolicyValueNet, model_config_from_checkpoint

    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = PolicyValueNet(model_config_from_checkpoint(ckpt))
    net.load_state_dict(ckpt["model"], strict=False)
    net.to(dev).eval()

    while not stop_event.is_set():
        try:
            first = request_q.get(timeout=REQUEST_TIMEOUT_S)
        except queue_mod.Empty:
            continue
        batch = [first]
        rows = first[2].shape[0]
        while rows < max_batch:
            try:
                nxt = request_q.get_nowait()
            except queue_mod.Empty:
                break
            batch.append(nxt)
            rows += nxt[2].shape[0]

        width = max(item[5].shape[1] for item in batch)
        hex_parts, glob_parts, fac_parts, cand_parts, mask_parts = [], [], [], [], []
        for item in batch:
            _wid, _rid, hex_np, glob_np, fac_np, cand_np, mask_np = item
            n, w = cand_np.shape[0], cand_np.shape[1]
            if w < width:
                cand_np = np.pad(cand_np, ((0, 0), (0, width - w), (0, 0)))
                mask_np = np.pad(mask_np, ((0, 0), (0, width - w)))
            hex_parts.append(hex_np)
            glob_parts.append(glob_np)
            fac_parts.append(fac_np)
            cand_parts.append(cand_np)
            mask_parts.append(mask_np)

        with torch.no_grad():
            logits, values = net(
                torch.from_numpy(np.concatenate(hex_parts)).to(dev),
                torch.from_numpy(np.concatenate(glob_parts)).to(dev),
                torch.from_numpy(np.concatenate(fac_parts)).to(dev),
                torch.from_numpy(np.concatenate(cand_parts)).to(dev),
                torch.from_numpy(np.concatenate(mask_parts)).to(dev),
            )
            priors = logits.softmax(dim=-1).cpu().numpy()
            values_np = values.cpu().numpy()

        offset = 0
        for item in batch:
            wid, rid = item[0], item[1]
            n, w = item[5].shape[0], item[5].shape[1]
            response_qs[wid].put(
                (rid, priors[offset : offset + n, :w].copy(), values_np[offset : offset + n].copy())
            )
            offset += n


class RemoteEvaluator:
    """Worker-side client: blocking round-trip to the inference server.

    Matches MCTSAgent's evaluator seam: called with pre-encoded numpy
    arrays, returns ``(priors_padded, values)`` for the whole batch.
    """

    def __init__(self, worker_id: int, request_q, response_q) -> None:
        self.worker_id = worker_id
        self.request_q = request_q
        self.response_q = response_q
        self._req = 0
        self.timeout_s = 60.0

    def __call__(
        self,
        hex_np: np.ndarray,
        glob_np: np.ndarray,
        fac_np: np.ndarray,
        cand_np: np.ndarray,
        mask_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._req += 1
        rid = self._req
        self.request_q.put((self.worker_id, rid, hex_np, glob_np, fac_np, cand_np, mask_np))
        while True:
            try:
                got_rid, priors, values = self.response_q.get(timeout=self.timeout_s)
            except queue_mod.Empty as exc:
                raise RuntimeError(
                    f"inference server unresponsive for {self.timeout_s}s "
                    f"(worker {self.worker_id}, request {rid}) -- server dead?"
                ) from exc
            if got_rid == rid:
                return priors, values
            # stale response from a cancelled request; drop and keep waiting


def start_server(checkpoint: Path, device: str, n_workers: int, max_batch: int = 256):
    """Spawn the server process; returns (process, request_q, response_qs, stop_event)."""
    from multiprocessing import get_context

    ctx = get_context("spawn")
    request_q = ctx.Queue()
    response_qs = {i: ctx.Queue() for i in range(n_workers)}
    stop_event = ctx.Event()
    proc = ctx.Process(
        target=serve,
        args=(str(checkpoint), device, request_q, response_qs, stop_event, max_batch),
        daemon=True,
    )
    proc.start()
    return proc, request_q, response_qs, stop_event
