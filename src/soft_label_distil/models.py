from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TopLogProbDTO:
    token: str
    logprob: float


@dataclass(frozen=True, slots=True)
class LogProbDTO:
    token: str
    logprob: float
    top_logprobs: list[TopLogProbDTO]
