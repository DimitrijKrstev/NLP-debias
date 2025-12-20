from pydantic import BaseModel


class SentenceWithRank(BaseModel):
    id: int
    rank: int
    text: str

    def __str__(self) -> str:
        return f"id: {self.id} rank: {self.rank} text: {self.text}\n"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SentenceWithRank):
            return False
        return self.id == other.id and self.text == other.text

    def __hash__(self) -> int:
        return hash((self.id, self.text))


class ModelPreference(BaseModel):
    sentences: list[SentenceWithRank]
    overall_reasoning: str | None
