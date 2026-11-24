"""User management commands."""

import typer

from cli import client
from cli.output import (
    display_detail,
    display_list,
    output_option,
    set_format_override,
)

app = typer.Typer(no_args_is_help=True)

USER_LIST_COLUMNS = ["ID", "Username", "Email", "Role", "Active", "Org"]
USER_LIST_KEYS = ["id", "username", "email", "role", "is_active", "organization_id"]

USER_DETAIL_FIELDS = [
    ("ID", "id"),
    ("Username", "username"),
    ("Email", "email"),
    ("Role", "role"),
    ("Active", "is_active"),
    ("Organization", "organization_id"),
    ("Create CA", "can_create_ca"),
    ("Create Cert", "can_create_cert"),
    ("Revoke Cert", "can_revoke_cert"),
    ("Export Key", "can_export_private_key"),
    ("Delete CA", "can_delete_ca"),
    ("Created At", "created_at"),
    ("Updated At", "updated_at"),
]


def _callback(output: str | None = output_option()) -> None:
    set_format_override(output)


app.callback(invoke_without_command=True)(_callback)


@app.command("list")
def list_users(
    skip: int = typer.Option(0, "--skip"),
    limit: int = typer.Option(100, "--limit"),
) -> None:
    """List all users (superuser only)."""
    data = client.get("/api/v1/users/", params={"skip": skip, "limit": limit}).json()
    display_list(data, USER_LIST_COLUMNS, keys=USER_LIST_KEYS, title="Users")


@app.command()
def me() -> None:
    """Show current user details."""
    data = client.get("/api/v1/users/me").json()
    display_detail(data, USER_DETAIL_FIELDS, title="Current User")


@app.command()
def show(user_id: int = typer.Argument(..., help="User ID")) -> None:
    """Show user details."""
    data = client.get(f"/api/v1/users/{user_id}").json()
    display_detail(data, USER_DETAIL_FIELDS, title=f"User #{user_id}")


@app.command()
def create(
    username: str = typer.Option(..., "--username", "-u", prompt=True),
    email: str = typer.Option(..., "--email", "-e", prompt=True),
    password: str = typer.Option(..., "--password", "-p", prompt=True, hide_input=True),
    role: str = typer.Option(
        "user", "--role", "-r", help="Role: user, admin, superuser"
    ),
    org_id: int | None = typer.Option(
        None,
        "--org",
        help="Organization ID (superusers: any org; admins: own org, the default)",
    ),
    can_create_ca: bool = typer.Option(False, "--can-create-ca"),
    can_create_cert: bool = typer.Option(False, "--can-create-cert"),
    can_revoke_cert: bool = typer.Option(False, "--can-revoke-cert"),
    can_export_private_key: bool = typer.Option(False, "--can-export-key"),
    can_delete_ca: bool = typer.Option(False, "--can-delete-ca"),
) -> None:
    """Create a new user.

    Superusers can create any role in any organization. Admins can create
    'user' accounts in their own organization only. Regular users cannot
    create accounts.
    """
    payload: dict[str, object] = {
        "username": username,
        "email": email,
        "password": password,
        "role": role,
        "can_create_ca": can_create_ca,
        "can_create_cert": can_create_cert,
        "can_revoke_cert": can_revoke_cert,
        "can_export_private_key": can_export_private_key,
        "can_delete_ca": can_delete_ca,
    }
    if org_id is not None:
        payload["organization_id"] = org_id

    data = client.post("/api/v1/users/", json=payload).json()
    display_detail(data, USER_DETAIL_FIELDS, title="User Created")


@app.command()
def update(
    user_id: int = typer.Argument(..., help="User ID"),
    email: str | None = typer.Option(None, "--email", "-e"),
    password: str | None = typer.Option(None, "--password", "-p"),
    role: str | None = typer.Option(None, "--role", "-r"),
    active: bool | None = typer.Option(None, "--active/--inactive"),
    org_id: int | None = typer.Option(None, "--org"),
    can_create_ca: bool | None = typer.Option(None, "--can-create-ca/--no-create-ca"),
    can_create_cert: bool | None = typer.Option(
        None, "--can-create-cert/--no-create-cert"
    ),
    can_revoke_cert: bool | None = typer.Option(
        None, "--can-revoke-cert/--no-revoke-cert"
    ),
    can_export_private_key: bool | None = typer.Option(
        None, "--can-export-key/--no-export-key"
    ),
    can_delete_ca: bool | None = typer.Option(None, "--can-delete-ca/--no-delete-ca"),
) -> None:
    """Update a user."""
    payload: dict[str, object] = {}
    if email is not None:
        payload["email"] = email
    if password is not None:
        payload["password"] = password
    if role is not None:
        payload["role"] = role
    if active is not None:
        payload["is_active"] = active
    if org_id is not None:
        payload["organization_id"] = org_id
    if can_create_ca is not None:
        payload["can_create_ca"] = can_create_ca
    if can_create_cert is not None:
        payload["can_create_cert"] = can_create_cert
    if can_revoke_cert is not None:
        payload["can_revoke_cert"] = can_revoke_cert
    if can_export_private_key is not None:
        payload["can_export_private_key"] = can_export_private_key
    if can_delete_ca is not None:
        payload["can_delete_ca"] = can_delete_ca

    if not payload:
        typer.echo("Nothing to update. Provide at least one flag.")
        raise typer.Exit(1)

    data = client.patch(f"/api/v1/users/{user_id}", json=payload).json()
    display_detail(data, USER_DETAIL_FIELDS, title=f"User #{user_id} Updated")


@app.command()
def delete(
    user_id: int = typer.Argument(..., help="User ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a user (superuser only)."""
    if not force:
        typer.confirm(f"Delete user #{user_id}?", abort=True)
    client.delete(f"/api/v1/users/{user_id}")
    typer.echo(f"User #{user_id} deleted.")
