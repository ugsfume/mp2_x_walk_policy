"""Mini Pupper V2 forward-walking policies as an Isaac Lab external task package.

Two training tasks and their evaluation variants are registered by
:func:`register_tasks`, which the Isaac Lab trainer calls through
``--external_callback mp2_x_walk_policy.register_tasks``.
"""

from __future__ import annotations


def register_tasks() -> None:
    """Register the gym ids. Returning ``None`` (not ``[]``) matters: the
    Isaac Lab trainer intersects its remaining CLI tokens with this return
    value, and an empty list silently discards every ``env.*`` / ``agent.*``
    override."""
    from . import registry  # noqa: F401  (registration happens on import)

    return None
