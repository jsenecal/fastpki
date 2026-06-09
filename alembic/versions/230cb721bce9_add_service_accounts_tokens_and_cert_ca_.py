"""add service accounts, tokens, and cert/ca ownership

Revision ID: 230cb721bce9
Revises: b978e2eebd16
Create Date: 2026-06-08 14:40:18.494344

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "230cb721bce9"
down_revision: str | Sequence[str] | None = "b978e2eebd16"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "service_accounts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("can_create_ca", sa.Boolean(), nullable=False),
        sa.Column("can_create_cert", sa.Boolean(), nullable=False),
        sa.Column("can_revoke_cert", sa.Boolean(), nullable=False),
        sa.Column("can_export_private_key", sa.Boolean(), nullable=False),
        sa.Column("can_delete_ca", sa.Boolean(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id", "name", name="uq_service_account_org_name"
        ),
    )
    with op.batch_alter_table("service_accounts", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_service_accounts_name"), ["name"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_service_accounts_organization_id"),
            ["organization_id"],
            unique=False,
        )

    op.create_table(
        "service_account_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("service_account_id", sa.Integer(), nullable=False),
        sa.Column("public_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("digest", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("pepper_version", sa.Integer(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["service_account_id"], ["service_accounts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("service_account_tokens", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_service_account_tokens_public_id"),
            ["public_id"],
            unique=True,
        )
        batch_op.create_index(
            batch_op.f("ix_service_account_tokens_service_account_id"),
            ["service_account_id"],
            unique=False,
        )

    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("service_account_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "service_account_name",
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=True,
            )
        )
        batch_op.create_index(
            batch_op.f("ix_audit_logs_service_account_id"),
            ["service_account_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            "fk_audit_service_account_id",
            "service_accounts",
            ["service_account_id"],
            ["id"],
        )

    with op.batch_alter_table("certificate_authorities", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("created_by_service_account_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_ca_created_by_service_account_id",
            "service_accounts",
            ["created_by_service_account_id"],
            ["id"],
        )

    with op.batch_alter_table("certificates", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("created_by_service_account_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_cert_created_by_service_account_id",
            "service_accounts",
            ["created_by_service_account_id"],
            ["id"],
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("certificates", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_cert_created_by_service_account_id", type_="foreignkey"
        )
        batch_op.drop_column("created_by_service_account_id")

    with op.batch_alter_table("certificate_authorities", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_ca_created_by_service_account_id", type_="foreignkey"
        )
        batch_op.drop_column("created_by_service_account_id")

    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.drop_constraint("fk_audit_service_account_id", type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_audit_logs_service_account_id"))
        batch_op.drop_column("service_account_name")
        batch_op.drop_column("service_account_id")

    with op.batch_alter_table("service_account_tokens", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_service_account_tokens_service_account_id"))
        batch_op.drop_index(batch_op.f("ix_service_account_tokens_public_id"))
    op.drop_table("service_account_tokens")

    with op.batch_alter_table("service_accounts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_service_accounts_organization_id"))
        batch_op.drop_index(batch_op.f("ix_service_accounts_name"))
    op.drop_table("service_accounts")
