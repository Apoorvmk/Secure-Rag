"""Shared server-side role and document access policy."""

ROLE_LEVELS = {"intern": 0, "employee": 1, "manager": 2, "board": 3}
ROLES = frozenset(ROLE_LEVELS)
TOP_K = 3


def allowed_access_levels(role: str) -> list[str]:
    """Return the levels visible to a known role; reject malformed roles."""
    if role not in ROLE_LEVELS:
        raise ValueError("Unknown user role")
    return [name for name, level in ROLE_LEVELS.items() if level <= ROLE_LEVELS[role]]


def can_upload_to_level(role: str, access_level: str) -> bool:
    """Upload policy follows the existing visibility hierarchy.

    A role can create documents at or below its own access level. This allows
    an employee to upload intern or employee material, while preventing an
    intern from assigning a privileged tier.
    """
    return role in ROLE_LEVELS and access_level in ROLE_LEVELS and ROLE_LEVELS[access_level] <= ROLE_LEVELS[role]
