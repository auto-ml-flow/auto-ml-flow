from enum import Enum


class Status(str, Enum):
    CREATED = "CREATED"
    STARTED = "STARTED"
    FAILED = "FAILED"
    DONE = "DONE"