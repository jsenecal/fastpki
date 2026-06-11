import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from app.core.config import settings
from app.db.models import Certificate, CertificateAuthority, CRLEntry
from app.services.encryption import EncryptionService
from app.services.exceptions import HasDependentsError, NotFoundError

UTC = ZoneInfo("UTC")


class CAService:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def generate_key_pair(key_size: int = 2048) -> tuple[rsa.RSAPrivateKey, bytes]:
        """Generate RSA key pair with key object and PEM-encoded private key."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )

        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        return private_key, private_key_pem

    @staticmethod
    def parse_subject_dn(subject_dn: str) -> x509.Name:
        """Parse a subject DN string into a cryptography Name object."""
        parts = {}
        for part in re.split(r"(?<!\\),", subject_dn):
            key, value = part.strip().split("=", 1)
            parts[key.strip()] = value.strip().replace("\\,", ",")

        name_attributes = []

        if "CN" in parts:
            name_attributes.append(x509.NameAttribute(NameOID.COMMON_NAME, parts["CN"]))
        if "O" in parts:
            name_attributes.append(
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, parts["O"])
            )
        if "OU" in parts:
            name_attributes.append(
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, parts["OU"])
            )
        if "C" in parts:
            name_attributes.append(x509.NameAttribute(NameOID.COUNTRY_NAME, parts["C"]))
        if "ST" in parts:
            name_attributes.append(
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, parts["ST"])
            )
        if "L" in parts:
            name_attributes.append(
                x509.NameAttribute(NameOID.LOCALITY_NAME, parts["L"])
            )

        return x509.Name(name_attributes)

    async def create_ca(
        self,
        name: str,
        subject_dn: str,
        description: str | None = None,
        key_size: int | None = None,
        valid_days: int | None = None,
        organization_id: int | None = None,
        created_by_user_id: int | None = None,
        created_by_service_account_id: int | None = None,
        parent_ca_id: int | None = None,
        path_length: int | None = None,
        allow_leaf_certs: bool | None = None,
        crl_base_url: str | None = None,
        base_url: str | None = None,
    ) -> CertificateAuthority:
        """Create a new Certificate Authority (root or intermediate)."""
        key_size = key_size or settings.CA_KEY_SIZE
        valid_days = valid_days or settings.CA_CERT_DAYS

        # Generate key pair
        private_key, private_key_pem = CAService.generate_key_pair(key_size)

        # Parse subject DN
        subject = CAService.parse_subject_dn(subject_dn)

        now = datetime.now(UTC)
        cert_builder = x509.CertificateBuilder()
        cert_builder = cert_builder.subject_name(subject)
        cert_builder = cert_builder.not_valid_before(now)
        cert_builder = cert_builder.not_valid_after(now + timedelta(days=valid_days))
        cert_builder = cert_builder.serial_number(x509.random_serial_number())
        cert_builder = cert_builder.public_key(private_key.public_key())

        parent_ca: CertificateAuthority | None = None
        if parent_ca_id is not None:
            # Intermediate CA: signed by parent
            parent_ca = await self.db.get(CertificateAuthority, parent_ca_id)
            if not parent_ca:
                raise ValueError(f"Parent CA with ID {parent_ca_id} not found")  # noqa: TRY003

            # Load parent private key and certificate
            parent_key_pem = EncryptionService.decrypt_private_key(
                parent_ca.private_key
            )
            parent_private_key = serialization.load_pem_private_key(
                parent_key_pem.encode("utf-8"),
                password=None,
            )
            parent_cert = x509.load_pem_x509_certificate(
                parent_ca.certificate.encode("utf-8"),
            )

            # Validate parent's BasicConstraints
            try:
                bc_ext = parent_cert.extensions.get_extension_for_class(
                    x509.BasicConstraints
                )
                bc = bc_ext.value
            except x509.ExtensionNotFound:
                raise ValueError("Parent CA certificate does not have BasicConstraints")  # noqa: B904, TRY003

            if not bc.ca:
                raise ValueError("Parent certificate is not a CA")  # noqa: TRY003

            if bc.path_length is not None and bc.path_length < 1:
                raise ValueError(  # noqa: TRY003
                    f"Parent CA path_length ({bc.path_length}) does not allow sub-CAs"
                )

            # Calculate effective path_length for the new intermediate
            effective_path_length: int | None
            if bc.path_length is not None:
                max_child_path_length = bc.path_length - 1
                if path_length is not None:
                    if path_length > max_child_path_length:
                        raise ValueError(  # noqa: TRY003
                            f"Requested path_length ({path_length}) exceeds maximum "
                            f"allowed by parent ({max_child_path_length})"
                        )
                    effective_path_length = path_length
                else:
                    effective_path_length = max_child_path_length
            else:
                effective_path_length = path_length

            # Set issuer from parent
            cert_builder = cert_builder.issuer_name(parent_cert.subject)
            signing_key = parent_private_key
        else:
            # Root CA: self-signed
            cert_builder = cert_builder.issuer_name(subject)
            signing_key = private_key
            effective_path_length = path_length

        # Add CA extensions
        cert_builder = cert_builder.add_extension(
            x509.BasicConstraints(ca=True, path_length=effective_path_length),
            critical=True,
        )
        cert_builder = cert_builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        cert_builder = cert_builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()),
            critical=False,
        )

        # Add AuthorityKeyIdentifier for intermediate CAs
        if parent_ca_id is not None:
            parent_ski = parent_cert.extensions.get_extension_for_class(
                x509.SubjectKeyIdentifier
            )
            cert_builder = cert_builder.add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(
                    parent_ski.value
                ),
                critical=False,
            )

            # Add CDP and AIA extensions pointing to parent CA's public URLs
            if base_url is not None and parent_ca is not None:
                parent_urls = CAService.get_public_urls(parent_ca, base_url)
                cert_builder = cert_builder.add_extension(
                    x509.CRLDistributionPoints(
                        [
                            x509.DistributionPoint(
                                full_name=[
                                    x509.UniformResourceIdentifier(parent_urls["crl"])
                                ],
                                relative_name=None,
                                crl_issuer=None,
                                reasons=None,
                            )
                        ]
                    ),
                    critical=False,
                )
                cert_builder = cert_builder.add_extension(
                    x509.AuthorityInformationAccess(
                        [
                            x509.AccessDescription(
                                x509.oid.AuthorityInformationAccessOID.CA_ISSUERS,
                                x509.UniformResourceIdentifier(parent_urls["ca_cert"]),
                            )
                        ]
                    ),
                    critical=False,
                )

        # Sign the certificate
        certificate = cert_builder.sign(
            private_key=signing_key,
            algorithm=hashes.SHA256(),
        )

        # Encode certificate to PEM
        certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)

        # Determine allow_leaf_certs value
        effective_allow_leaf_certs = (
            allow_leaf_certs if allow_leaf_certs is not None else True
        )

        # Create CA database entry
        ca = CertificateAuthority(
            name=name,
            description=description,
            subject_dn=subject_dn,
            key_size=key_size,
            valid_days=valid_days,
            private_key=EncryptionService.encrypt_private_key(
                private_key_pem.decode("utf-8")
            ),
            certificate=certificate_pem.decode("utf-8"),
            organization_id=organization_id,
            created_by_user_id=created_by_user_id,
            created_by_service_account_id=created_by_service_account_id,
            parent_ca_id=parent_ca_id,
            path_length=effective_path_length,
            allow_leaf_certs=effective_allow_leaf_certs,
            crl_base_url=crl_base_url,
        )

        self.db.add(ca)

        # Auto-set parent's allow_leaf_certs to False when creating an intermediate CA
        if parent_ca is not None:
            parent_ca.allow_leaf_certs = False

        await self.db.commit()
        await self.db.refresh(ca)

        return ca

    async def get_ca(self, ca_id: int) -> CertificateAuthority | None:
        """Get a Certificate Authority by ID."""
        ca = await self.db.get(CertificateAuthority, ca_id)
        return ca

    async def get_ca_by_name(
        self, name: str, organization_id: int | None = None
    ) -> CertificateAuthority | None:
        """Get a CA by name, optionally scoped to an organization."""
        query = select(CertificateAuthority).where(CertificateAuthority.name == name)
        if organization_id is not None:
            query = query.where(CertificateAuthority.organization_id == organization_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_cas(
        self, organization_id: int | None = None
    ) -> list[CertificateAuthority]:
        """List Certificate Authorities, optionally filtered by organization."""
        query = select(CertificateAuthority)
        if organization_id is not None:
            query = query.where(CertificateAuthority.organization_id == organization_id)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def assign_organization(
        self, ca_id: int, organization_id: int, cascade: bool = False
    ) -> tuple[CertificateAuthority, int, int]:
        """Assign a CA to an organization.

        With cascade=True, also adopts all descendant CAs and every certificate
        issued by the affected CAs (intended for migrating pre-organization
        instances). Returns (ca, cas_updated, certs_updated).
        """
        ca = await self.db.get(CertificateAuthority, ca_id)
        if not ca:
            raise NotFoundError(f"Certificate Authority with ID {ca_id} not found")  # noqa: TRY003

        ca_ids = [ca_id]
        if cascade:
            queue = [ca_id]
            while queue:
                for child in await self.get_child_cas(queue.pop()):
                    if child.id is not None:
                        ca_ids.append(child.id)
                        queue.append(child.id)

        await self.db.execute(
            update(CertificateAuthority)
            .where(col(CertificateAuthority.id).in_(ca_ids))
            .values(organization_id=organization_id)
        )
        certs_updated = 0
        if cascade:
            result = await self.db.execute(
                update(Certificate)
                .where(col(Certificate.issuer_id).in_(ca_ids))
                .values(organization_id=organization_id)
            )
            certs_updated = result.rowcount  # type: ignore[attr-defined]

        await self.db.commit()
        await self.db.refresh(ca)
        return ca, len(ca_ids), certs_updated

    async def delete_ca(self, ca_id: int) -> bool:
        """Delete a Certificate Authority by ID.

        Raises HasDependentsError if the CA has child CAs.
        """
        ca = await self.db.get(CertificateAuthority, ca_id)
        if not ca:
            return False

        # Check for child CAs
        children = await self.get_child_cas(ca_id)
        if children:
            raise HasDependentsError(  # noqa: TRY003
                f"CA {ca_id} has {len(children)} child CA(s) and cannot be deleted"
            )

        await self.db.delete(ca)
        await self.db.commit()
        return True

    async def get_ca_chain(self, ca_id: int) -> list[CertificateAuthority]:
        """Get the CA chain from the given CA up to the root.

        Returns an ordered list starting with the given CA and ending at the root.
        """
        chain: list[CertificateAuthority] = []
        current_id: int | None = ca_id
        while current_id is not None:
            ca = await self.db.get(CertificateAuthority, current_id)
            if not ca:
                break
            chain.append(ca)
            current_id = ca.parent_ca_id
        return chain

    async def get_child_cas(self, ca_id: int) -> list[CertificateAuthority]:
        """Get direct child CAs of the given CA."""
        query = select(CertificateAuthority).where(
            CertificateAuthority.parent_ca_id == ca_id
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    def slugify(name: str) -> str:
        """Convert a CA name to a URL-safe slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug.strip("-")

    @staticmethod
    def get_slug(ca: CertificateAuthority) -> str:
        """Get the full slug for a CA: {name-slug}-{id}."""
        return f"{CAService.slugify(ca.name)}-{ca.id}"

    @staticmethod
    def parse_slug(slug: str) -> tuple[str, int]:
        """Parse a slug into (name_slug, ca_id). Raises ValueError on bad format."""
        last_dash = slug.rfind("-")
        if last_dash < 0:
            raise ValueError(f"Invalid slug: {slug}")  # noqa: TRY003
        name_part = slug[:last_dash]
        try:
            ca_id = int(slug[last_dash + 1 :])
        except ValueError:
            raise ValueError(f"Invalid slug: {slug}") from None  # noqa: TRY003
        return name_part, ca_id

    async def get_ca_by_slug(self, slug: str) -> CertificateAuthority | None:
        """Look up a CA by slug, validating the name prefix matches."""
        try:
            name_part, ca_id = CAService.parse_slug(slug)
        except ValueError:
            return None
        ca = await self.db.get(CertificateAuthority, ca_id)
        if ca is None or CAService.slugify(ca.name) != name_part:
            return None
        return ca

    @staticmethod
    def get_public_urls(ca: CertificateAuthority, base_url: str) -> dict[str, str]:
        """Get the public CDP and AIA URLs for a CA."""
        slug = CAService.get_slug(ca)
        base = (ca.crl_base_url or base_url).rstrip("/")
        return {
            "crl": f"{base}/crl/{slug}",
            "ca_cert": f"{base}/ca/{slug}.crt",
        }

    async def generate_crl(self, ca_id: int) -> bytes:
        """Generate a signed CRL in DER format for the given CA."""
        ca = await self.db.get(CertificateAuthority, ca_id)
        if not ca:
            raise ValueError(f"No CA: {ca_id}")  # noqa: TRY003

        # Load CA private key and certificate
        ca_key_pem = EncryptionService.decrypt_private_key(ca.private_key)
        ca_private_key = serialization.load_pem_private_key(
            ca_key_pem.encode("utf-8"),
            password=None,
        )
        ca_cert = x509.load_pem_x509_certificate(
            ca.certificate.encode("utf-8"),
        )

        # Query CRL entries for this CA
        query = select(CRLEntry).where(CRLEntry.ca_id == ca_id)
        result = await self.db.execute(query)
        crl_entries = list(result.scalars().all())

        # Build CRL
        now = datetime.now(UTC)
        crl_builder = x509.CertificateRevocationListBuilder()
        crl_builder = crl_builder.issuer_name(ca_cert.subject)
        crl_builder = crl_builder.last_update(now)
        crl_builder = crl_builder.next_update(now + timedelta(days=7))

        for entry in crl_entries:
            revoked = (
                x509.RevokedCertificateBuilder()
                .serial_number(int(entry.serial_number, 16))
                .revocation_date(entry.revocation_date)
                .build()
            )
            crl_builder = crl_builder.add_revoked_certificate(revoked)

        crl = crl_builder.sign(
            private_key=ca_private_key,
            algorithm=hashes.SHA256(),
        )

        return crl.public_bytes(serialization.Encoding.DER)
