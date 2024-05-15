import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union
from urllib.parse import urljoin

import requests
from pydantic import TypeAdapter, ValidationError
from tenacity import (
    RetryError,
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from auto_ml_flow.client.exceptions import BaseURLNotProvided, ClientBadRequestError, ClientConnectionError, ClientNotFoundError, ClientServerError, ClientValidationError
from loguru import logger

T = TypeVar('T')


class BaseClient:
    def __init__(self, base_url: Optional[str] = None, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        self.base_url = base_url

        if not self.base_url:
            raise BaseURLNotProvided('Not provided default url')

    @retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.WARN),
        retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    )
    def _request(
        self,
        path: str,
        *,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        stream: bool = False,
        timeout: int = 100
    ):
        """
        Выполнить запрос по указанному пути с заданным методом и параметрами.

        :param path: Путь для запроса, относительный к базовому URL.
        :param method: Используемый HTTP-метод (например, GET, POST, PUT, DELETE).
        :param params: Необязательный словарь параметров запроса.
        :param json: Необязательный словарь или JSON-сериализуемые данные для отправки в теле запроса (для POST, PUT).
        :param data: Необязательные данные для отправки в теле запроса (для POST, PUT).
        :param stream: Следует ли передавать содержимое ответа по потоку или нет.
        :param timeout: Значение тайм-аута для запроса в секундах.
        :return: Объект ответа.
        """

        url = urljoin(self.base_url, path)  # Объединить базовый URL и путь

        resp = self.session.request(method, url, params=params, json=json, data=data, stream=stream, timeout=timeout)
        resp.raise_for_status()

        return resp

    def _make_request(
        self,
        path: str,
        method: str,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
    ) -> Union[T, Any]:
        """
        Выполнить запрос и вернуть экземпляр указанной модели.

        :param path: Путь для запроса, относительный к базовому URL.
        :param method: Используемый HTTP-метод (например, GET, POST, PUT, DELETE).
        :param model: Необязательный класс модели для создания экземпляра из данных ответа (если не передан - просто возвращаем результат).
        :param params: Необязательный словарь параметров запроса.
        :param json: Необязательный словарь или JSON-сериализуемые данные для отправки в теле запроса (для POST, PUT).
        :param data: Необязательные данные для отправки в теле запроса (для POST, PUT).
        :return: Экземпляр указанной модели.
        """

        try:
            resp = self._request(path, method=method, params=params, json=json, data=data)
            resp.raise_for_status()
        
        except (requests.exceptions.ConnectionError, RetryError) as err:
            raise ClientConnectionError from err
        
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            
            if status_code == 400:
                raise ClientBadRequestError(http_err.response.json()) from http_err
            elif status_code == 404:
                raise ClientNotFoundError(http_err.response.json()) from http_err
            elif status_code >= 500:
                raise ClientServerError(http_err.response.json()) from http_err
            else:
                raise

        data = resp.json()

        if not model:
            return data

        type_adapted_model = TypeAdapter(model)

        try:
            return type_adapted_model.validate_python(data)
        except ValidationError as err:
            raise ClientValidationError from err

    def _get(self, path: str, *, model: Type[T], params: Optional[Dict[str, Any]] = None) -> Union[T, Any]:
        return self._make_request(path, "GET", model, params=params)

    def _post(
        self,
        path: str,
        *,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None
    ) -> Union[T, Any]:
        return self._make_request(path, "POST", model, params=params, json=json, data=data)

    def _put(
        self,
        path: str,
        *,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None
    ) -> Union[T, Any]:
        return self._make_request(path, "PUT", model, params=params, json=json, data=data)

    def _patch(
        self,
        path: str,
        *,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None
    ) -> Union[T, Any]:
        return self._make_request(path, "PATCH", model, params=params, json=json, data=data)

    def _delete(
        self, path: str, *, model: Optional[Type[T]] = None, params: Optional[Dict[str, Any]] = None
    ) -> Union[T, Any]:
        return self._make_request(path, "DELETE", model, params=params)
