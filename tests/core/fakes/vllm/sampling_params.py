import enum


class RequestOutputKind(enum.Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2
