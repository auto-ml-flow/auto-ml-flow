from http import HTTPStatus
from typing import Any, Dict, Optional, Type, TypeVar
from urllib.parse import urljoin

import requests
from pydantic import TypeAdapter, ValidationError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from auto_ml_flow.client.exceptions import (
    BaseURLNotProvidedError,
    ClientBadRequestError,
    ClientConnectionError,
    ClientNotFoundError,
    ClientServerError,
    ClientValidationError,
)

T = TypeVar("T")


class BaseClient:
    def __init__(
        self, base_url: Optional[str] = None, session: Optional[requests.Session] = None
    ) -> None:
        self.session = session or requests.Session()
        self.base_url = base_url

        if not self.base_url:
            raise BaseURLNotProvidedError("Not provided default url")

    @retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    )
    def _request(
        self,
        path: str,
        *,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: int = 100,
    ) -> requests.Response:
        if not self.base_url:
            raise BaseURLNotProvidedError("Base URL not provided")

        url = urljoin(self.base_url, path)  # Объединить базовый URL и путь

        resp = self.session.request(
            method, url, params=params, json=json, data=data, stream=stream, timeout=timeout
        )
        resp.raise_for_status()

        return resp

    def _make_request(
        self,
        path: str,
        method: str,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> T:
        try:
            resp = self._request(path, method=method, params=params, json=json, data=data)
            resp.raise_for_status()

        except (requests.exceptions.ConnectionError, RetryError) as err:
            raise ClientConnectionError from err

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code

            if status_code == HTTPStatus.BAD_REQUEST.value:
                raise ClientBadRequestError(http_err.response.json()) from http_err

            if status_code == HTTPStatus.NOT_FOUND.value:
                raise ClientNotFoundError(http_err.response.json()) from http_err

            if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR.value:
                raise ClientServerError(http_err.response.json()) from http_err

            raise

        data = resp.json()

        if not model:
            return data  # type: ignore

        type_adapted_model = TypeAdapter(model)

        try:
            return type_adapted_model.validate_python(data)
        except ValidationError as err:
            raise ClientValidationError from err

    def _get(self, path: str, *, model: Type[T], params: Optional[Dict[str, Any]] = None) -> T:
        return self._make_request(path, "GET", model, params=params)

    def _post(
        self,
        path: str,
        *,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> T:
        return self._make_request(path, "POST", model, params=params, json=json, data=data)

    def _put(
        self,
        path: str,
        *,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> T:
        return self._make_request(path, "PUT", model, params=params, json=json, data=data)

    def _patch(
        self,
        path: str,
        *,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> T:
        return self._make_request(path, "PATCH", model, params=params, json=json, data=data)

    def _delete(
        self, path: str, *, model: Optional[Type[T]] = None, params: Optional[Dict[str, Any]] = None
    ) -> T:
        return self._make_request(path, "DELETE", model, params=params)
