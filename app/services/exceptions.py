class ServiceError(Exception):
    """Base exception for service-layer errors."""


class NotFoundError(ServiceError):
    """Raised when a requested resource is not found."""


class AlreadyExistsError(ServiceError):
    """Raised when attempting to create a resource that already exists."""


class PermissionDeniedError(ServiceError):
    """Raised when the user lacks permission for an operation."""


class HasDependentsError(ServiceError):
    """Raised when a resource cannot be deleted because it has dependents."""


class LeafCertNotAllowedError(ServiceError):
    """Raised when a CA does not allow leaf certificate issuance."""


class CsrRequiredError(ServiceError):
    """Raised when renewing a CSR-origin certificate without supplying a CSR."""


class CsrNotAllowedError(ServiceError):
    """Raised when supplying a CSR to renew a server-key certificate."""


class IssuancePolicyMissingError(ServiceError):
    """Raised when a service account attempts issuance with no policy attached."""


class PolicyViolationError(ServiceError):
    """Raised when an issuance request violates a service account's policy.

    Carries the offending policy ``field`` and ``value`` for a structured
    API error response.
    """

    def __init__(self, field: str, value: object) -> None:
        self.field = field
        self.value = value
        super().__init__(f"Issuance policy violation on '{field}': {value!r}")
