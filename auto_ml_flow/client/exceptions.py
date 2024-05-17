class ClientError(ConnectionError, Exception): ...


class ClientConnectionError(ClientError): ...


class ClientValidationError(ClientError): ...


class BaseURLNotProvidedError(ClientError): ...


class ClientBadRequestError(ClientError):
    """Exception raised for 400 Bad Request HTTP error."""


class ClientNotFoundError(ClientError):
    """Exception raised for 404 Not Found HTTP error."""


class ClientServerError(ClientError):
    """Exception raised for 500 Internal Server Error HTTP error."""
