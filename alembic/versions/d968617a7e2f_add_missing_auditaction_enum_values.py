"""Add missing auditaction enum values

Adds CA_UPDATE plus the SERVICE_ACCOUNT_* values that were introduced in
0.4.0 without a corresponding enum migration. Only PostgreSQL stores a
native enum type; on SQLite the column is a plain VARCHAR, so no change is
needed there.

Revision ID: d968617a7e2f
Revises: c7d99ce53025
Create Date: 2026-06-11 09:35:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d968617a7e2f"
down_revision: str | Sequence[str] | None = "c7d99ce53025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW_ACTIONS = (
    "CA_UPDATE",
    "SERVICE_ACCOUNT_CREATE",
    "SERVICE_ACCOUNT_UPDATE",
    "SERVICE_ACCOUNT_DELETE",
    "SERVICE_ACCOUNT_TOKEN_CREATE",
    "SERVICE_ACCOUNT_TOKEN_REVOKE",
)


def upgrade() -> None:
    """Upgrade schema."""
    if op.get_bind().dialect.name == "postgresql":
        for action in NEW_ACTIONS:
            op.execute(f"ALTER TYPE auditaction ADD VALUE IF NOT EXISTS '{action}'")


def downgrade() -> None:
    """Downgrade schema.

    PostgreSQL cannot drop enum values; existing rows may already use them.
    Leaving the values in place is harmless, so this is a no-op.
    """
