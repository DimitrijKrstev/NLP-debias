from attr import dataclass
from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class SentenceWithRank:
    id: int
    rank: int
    text: str

    def __str__(self) -> str:
        return f"- ID: {self.id} rank: {self.rank} text: {self.text}\n"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SentenceWithRank):
            return False
        return self.id == other.id and self.text == other.text

    def __hash__(self) -> int:
        return hash((self.id, self.text))


@dataclass(frozen=True, slots=True)
class PreferencePair:
    prompt: list[dict[str, str]]
    formatted_prompt: str
    chosen: str
    rejected: str


class ModelOutputSentenceWithRank(BaseModel):
    id: int
    rank: int


class ModelPreference(BaseModel):
    sentences: list[ModelOutputSentenceWithRank]
    overall_reasoning: str | None
