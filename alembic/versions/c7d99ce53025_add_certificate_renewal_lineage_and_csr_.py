"""add certificate renewal lineage and csr-origin marker

Revision ID: c7d99ce53025
Revises: 38722803ba15
Create Date: 2026-06-08 20:11:25.068385

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7d99ce53025"
down_revision: str | Sequence[str] | None = "38722803ba15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("certificates", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "is_csr_origin",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.add_column(sa.Column("renewed_from_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_cert_renewed_from_id", "certificates", ["renewed_from_id"], ["id"]
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("certificates", schema=None) as batch_op:
        batch_op.drop_constraint("fk_cert_renewed_from_id", type_="foreignkey")
        batch_op.drop_column("renewed_from_id")
        batch_op.drop_column("is_csr_origin")
