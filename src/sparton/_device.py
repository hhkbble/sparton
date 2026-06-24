"""Active Triton/torch device for Sparton (import-time diagnostic).

Resolved once and logged at DEBUG. Not part of the public API; the facade
imports this module only for the side-effecting log.
"""

import logging

import triton

logger = logging.getLogger("sparton")

DEVICE = triton.runtime.driver.active.get_active_torch_device()
logger.debug("Sparton using device: %s", DEVICE)
