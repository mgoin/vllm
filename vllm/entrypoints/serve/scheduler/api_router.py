# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime scheduler config endpoint.

Prototype endpoint that exposes a live-update interface for a whitelist of
``SchedulerConfig`` fields such as ``max_num_batched_tokens`` and
``max_num_seqs``. Gated behind ``VLLM_SERVER_DEV_MODE``.

Example:
    GET  /scheduler_config
    POST /scheduler_config
         {"max_num_batched_tokens": 4096, "max_num_seqs": 64}
"""

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/scheduler_config")
async def get_scheduler_config(raw_request: Request) -> JSONResponse:
    """Return the current values of live-updatable scheduler fields."""
    snapshot = await engine_client(raw_request).get_scheduler_config()
    return JSONResponse(content=snapshot)


@router.post("/scheduler_config")
async def update_scheduler_config(raw_request: Request) -> JSONResponse:
    """Live-update a subset of scheduler config fields.

    Body is a JSON object mapping field name to new value, e.g.::

        {"max_num_batched_tokens": 4096, "max_num_seqs": 64}

    Only whitelisted fields on ``SchedulerConfig`` can be updated at
    runtime. Updates are validated against ``max_model_len`` and rolled
    back atomically on failure.
    """
    try:
        updates: dict[str, Any] = await raw_request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON body: {exc}"
        ) from exc

    if not isinstance(updates, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object of field→value.",
        )

    logger.info("Live-updating scheduler config: %s", updates)
    try:
        snapshot = await engine_client(raw_request).update_scheduler_config(updates)
    except ValueError as exc:
        # Validation failures (invariant violations, unknown fields).
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Utility call-site wraps engine-side exceptions as generic errors;
        # surface them as 400 so callers see the reason.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(content=snapshot)


def attach_router(app: FastAPI) -> None:
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
