import typer

from cli.audit import app as audit_app
from cli.auth import app as auth_app
from cli.ca import app as ca_app
from cli.cert import app as cert_app
from cli.config_cmd import app as config_app
from cli.export import app as export_app
from cli.org import app as org_app
from cli.service_account import app as service_account_app
from cli.user import app as user_app

app = typer.Typer(
    name="fastpki",
    help="FastPKI CLI — API-based PKI management",
    no_args_is_help=True,
)

app.add_typer(auth_app, name="auth", help="Authentication (login/logout/status)")
app.add_typer(ca_app, name="ca", help="Certificate Authority management")
app.add_typer(cert_app, name="cert", help="Certificate management")
app.add_typer(org_app, name="org", help="Organization management")
app.add_typer(user_app, name="user", help="User management")
app.add_typer(
    service_account_app, name="service-account", help="Service account management"
)
app.add_typer(export_app, name="export", help="Export certificates and keys")
app.add_typer(audit_app, name="audit", help="Audit log queries")
app.add_typer(config_app, name="config", help="CLI configuration")


def main() -> None:
    app()
