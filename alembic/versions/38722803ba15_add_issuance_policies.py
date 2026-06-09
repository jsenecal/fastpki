"""add issuance policies

Revision ID: 38722803ba15
Revises: 230cb721bce9
Create Date: 2026-06-08 18:59:50.316109

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "38722803ba15"
down_revision: str | Sequence[str] | None = "230cb721bce9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "issuance_policies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("service_account_id", sa.Integer(), nullable=False),
        sa.Column("cn_patterns", sa.JSON(), nullable=True),
        sa.Column("san_dns_patterns", sa.JSON(), nullable=True),
        sa.Column("san_ip_cidrs", sa.JSON(), nullable=True),
        sa.Column("san_email_domains", sa.JSON(), nullable=True),
        sa.Column("allowed_ca_ids", sa.JSON(), nullable=True),
        sa.Column("allowed_certificate_types", sa.JSON(), nullable=True),
        sa.Column("max_validity_days", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["service_account_id"], ["service_accounts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("issuance_policies", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_issuance_policies_service_account_id"),
            ["service_account_id"],
            unique=True,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("issuance_policies", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_issuance_policies_service_account_id"))
    op.drop_table("issuance_policies")
