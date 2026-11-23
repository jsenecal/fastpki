import pytest
from cryptography import x509
from httpx import ASGITransport, AsyncClient

from app.db.models import CertificateType, UserRole
from app.db.session import get_session
from app.services.ca import CAService
from tests.conftest import get_test_session


@pytest.fixture
def public_client(setup_db):
    """Client that hits the full app (including /crl and /ca routes)."""
    from app.main import app

    app.dependency_overrides[get_session] = get_test_session
    return app


async def _bootstrap_superuser(client):
    """Create first superuser and return auth headers."""
    await client.post(
        "/api/v1/users/",
        json={
            "username": "super",
            "email": "super@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )
    login = await client.post(
        "/api/v1/auth/token",
        data={"username": "super", "password": "Kestrel-Meadow-4821"},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def _create_ca(client, headers, name="Test CA", parent_ca_id=None):
    """Create a CA and return the response data."""
    resp = await client.post(
        "/api/v1/cas/",
        json={
            "name": name,
            "subject_dn": f"CN={name},O=Test",
            "key_size": 2048,
            "valid_days": 365,
            "parent_ca_id": parent_ca_id,
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.mark.asyncio
async def test_crl_download_der_empty(public_client):
    """CRL download for a CA with no revocations returns a valid empty CRL."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _bootstrap_superuser(client)
        ca = await _create_ca(client, headers)
        slug = f"{CAService.slugify(ca['name'])}-{ca['id']}"

        resp = await client.get(f"/crl/{slug}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pkix-crl"

        crl = x509.load_der_x509_crl(resp.content)
        assert len(list(crl)) == 0


@pytest.mark.asyncio
async def test_crl_download_pem_empty(public_client):
    """CRL download in PEM format for a CA with no revocations."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _bootstrap_superuser(client)
        ca = await _create_ca(client, headers)
        slug = f"{CAService.slugify(ca['name'])}-{ca['id']}"

        resp = await client.get(f"/crl/{slug}.pem")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/x-pem-file"

        crl = x509.load_pem_x509_crl(resp.content)
        assert len(list(crl)) == 0


@pytest.mark.asyncio
async def test_crl_contains_revoked_serial(public_client):
    """After revoking a certificate, its serial appears in the CRL."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _bootstrap_superuser(client)
        ca = await _create_ca(client, headers)
        slug = f"{CAService.slugify(ca['name'])}-{ca['id']}"

        # Issue a certificate
        cert_resp = await client.post(
            f"/api/v1/certificates/?ca_id={ca['id']}",
            json={
                "common_name": "test.example.com",
                "subject_dn": "CN=test.example.com,O=Test",
                "certificate_type": CertificateType.SERVER.value,
                "key_size": 2048,
                "valid_days": 30,
            },
            headers=headers,
        )
        assert cert_resp.status_code == 201
        cert_data = cert_resp.json()
        serial_hex = cert_data["serial_number"]

        # Revoke it
        revoke_resp = await client.post(
            f"/api/v1/certificates/{cert_data['id']}/revoke",
            json={"reason": "testing"},
            headers=headers,
        )
        assert revoke_resp.status_code == 200

        # Download CRL and check
        crl_resp = await client.get(f"/crl/{slug}")
        assert crl_resp.status_code == 200
        crl = x509.load_der_x509_crl(crl_resp.content)

        revoked_serials = [format(entry.serial_number, "x") for entry in crl]
        assert serial_hex in revoked_serials


@pytest.mark.asyncio
async def test_ca_cert_download_der(public_client):
    """CA certificate download in DER format."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _bootstrap_superuser(client)
        ca = await _create_ca(client, headers)
        slug = f"{CAService.slugify(ca['name'])}-{ca['id']}"

        resp = await client.get(f"/ca/{slug}.crt")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pkix-cert"

        cert = x509.load_der_x509_certificate(resp.content)
        assert (
            cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
            == "Test CA"
        )


@pytest.mark.asyncio
async def test_ca_cert_download_pem(public_client):
    """CA certificate download in PEM format."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _bootstrap_superuser(client)
        ca = await _create_ca(client, headers)
        slug = f"{CAService.slugify(ca['name'])}-{ca['id']}"

        resp = await client.get(f"/ca/{slug}.pem")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/x-pem-file"
        assert resp.text == ca["certificate"]


@pytest.mark.asyncio
async def test_pki_404_nonexistent(public_client):
    """Nonexistent CA slug returns 404."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/crl/nonexistent-99999")
        assert resp.status_code == 404

        resp = await client.get("/ca/nonexistent-99999.crt")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_pki_404_wrong_slug_prefix(public_client):
    """Correct ID but wrong name prefix returns 404 (prevents enumeration)."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _bootstrap_superuser(client)
        ca = await _create_ca(client, headers)

        # Use correct ID but wrong name
        resp = await client.get(f"/crl/wrong-name-{ca['id']}")
        assert resp.status_code == 404

        resp = await client.get(f"/ca/wrong-name-{ca['id']}.crt")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_pki_endpoints_no_auth_required(public_client):
    """Public PKI endpoints work without any authentication."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Bootstrap with auth to create the CA
        headers = await _bootstrap_superuser(client)
        ca = await _create_ca(client, headers)
        slug = f"{CAService.slugify(ca['name'])}-{ca['id']}"

        # Access without any auth headers
        for url in [
            f"/crl/{slug}",
            f"/crl/{slug}.pem",
            f"/ca/{slug}.crt",
            f"/ca/{slug}.pem",
        ]:
            resp = await client.get(url)
            assert resp.status_code == 200, f"Failed for {url}: {resp.status_code}"


@pytest.mark.asyncio
async def test_cert_has_cdp_and_aia_extensions(public_client):
    """Issued certificates contain CDP and AIA extensions."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _bootstrap_superuser(client)
        ca = await _create_ca(client, headers)
        slug = f"{CAService.slugify(ca['name'])}-{ca['id']}"

        # Issue a certificate
        cert_resp = await client.post(
            f"/api/v1/certificates/?ca_id={ca['id']}",
            json={
                "common_name": "test.example.com",
                "subject_dn": "CN=test.example.com,O=Test",
                "certificate_type": CertificateType.SERVER.value,
                "key_size": 2048,
                "valid_days": 30,
            },
            headers=headers,
        )
        assert cert_resp.status_code == 201
        cert_pem = cert_resp.json()["certificate"]

        cert = x509.load_pem_x509_certificate(cert_pem.encode())

        # Check CDP extension
        cdp = cert.extensions.get_extension_for_class(x509.CRLDistributionPoints)
        cdp_urls = [name.value for dp in cdp.value for name in dp.full_name]
        assert any(slug in url for url in cdp_urls)

        # Check AIA extension
        aia = cert.extensions.get_extension_for_class(x509.AuthorityInformationAccess)
        aia_urls = [desc.access_location.value for desc in aia.value]
        assert any(slug in url for url in aia_urls)


@pytest.mark.asyncio
async def test_intermediate_ca_has_cdp_and_aia(public_client):
    """Intermediate CA certificates contain CDP and AIA pointing to parent."""
    transport = ASGITransport(app=public_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _bootstrap_superuser(client)
        root_ca = await _create_ca(client, headers, name="Root CA")
        root_slug = f"{CAService.slugify(root_ca['name'])}-{root_ca['id']}"

        inter_ca = await _create_ca(
            client, headers, name="Intermediate CA", parent_ca_id=root_ca["id"]
        )
        inter_cert_pem = inter_ca["certificate"]
        cert = x509.load_pem_x509_certificate(inter_cert_pem.encode())

        # CDP should point to root CA's CRL
        cdp = cert.extensions.get_extension_for_class(x509.CRLDistributionPoints)
        cdp_urls = [name.value for dp in cdp.value for name in dp.full_name]
        assert any(root_slug in url for url in cdp_urls)

        # AIA should point to root CA's cert
        aia = cert.extensions.get_extension_for_class(x509.AuthorityInformationAccess)
        aia_urls = [desc.access_location.value for desc in aia.value]
        assert any(root_slug in url for url in aia_urls)
