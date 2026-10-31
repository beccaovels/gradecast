"""Course structure and the deterministic (no-randomness) grade math.

A Course is a list of graded Items, each with a weight (its share of the final
grade) and optionally a score. This module answers plain questions such as
"what is my total so far?" and "what average do I need on the rest to get an
A?". It also holds the built-in course preset and an example set of scores.
"""

from dataclasses import dataclass, field
from typing import Literal

# The three kinds of item the predictors treat differently.
Kind = Literal["quiz", "exam", "other"]


@dataclass
class Item:
    """One graded component of a course.

    Attributes:
        name: Display name, e.g. "Quiz 1".
        weight: Percent of the whole course, e.g. 5.0 means 5%.
        kind: "quiz", "exam" or "other" (participation, exercises...).
        score: Score from 0-100, or None if not taken yet.
    """

    name: str
    weight: float
    kind: Kind
    score: float | None = None


@dataclass
class Course:
    """A course made of weighted items, with helper maths.

    Attributes:
        name: Course name.
        items: All graded items, taken or not.
    """

    name: str
    items: list[Item] = field(default_factory=list)

    def total_weight(self) -> float:
        """Sum of all item weights (should be 100).

        Example: items weighing 30 and 70 -> 100.0
        """
        return sum(item.weight for item in self.items)

    def taken(self, kind: Kind | None = None) -> list[Item]:
        """Items that already have a score, optionally filtered by kind.

        Args:
            kind: Keep only this kind, or None for every kind.
        """
        return [
            item
            for item in self.items
            if item.score is not None and (kind is None or item.kind == kind)
        ]

    def remaining(self, kind: Kind | None = None) -> list[Item]:
        """Items without a score yet, optionally filtered by kind.

        Args:
            kind: Keep only this kind, or None for every kind.
        """
        return [
            item
            for item in self.items
            if item.score is None and (kind is None or item.kind == kind)
        ]

    def current_total(self) -> float:
        """Points earned so far, out of 100.

        Example: a 5% item scored 80 contributes 80 * 5 / 100 = 4 points.
        """
        return sum(item.score * item.weight / 100 for item in self.taken())

    def remaining_weight(self) -> float:
        """Total weight (percent) of items not taken yet."""
        return sum(item.weight for item in self.remaining())

    def required_average(self, target: float = 85) -> float | None:
        """Average score needed on ALL remaining items to reach ``target``.

        Args:
            target: Course total you want, 85 = the A line.

        Returns:
            The needed average (can exceed 100 = out of reach, 0 = already
            secured), or None if nothing remains.
        """
        remaining = self.remaining_weight()
        if remaining == 0:
            return None
        # max(..., 0) so an already-secured target gives 0, not a negative.
        return max(target - self.current_total(), 0) / remaining * 100

    def best_case(self) -> float:
        """Highest possible final total: perfect scores on everything left."""
        return self.current_total() + self.remaining_weight()

    def validate(self) -> list[str]:
        """Check the course for mistakes a student might have typed.

        Returns:
            Human-readable warnings; an empty list means all good.
        """
        warnings: list[str] = []
        # Weights should add up to 100%, allowing tiny float rounding errors.
        if abs(self.total_weight() - 100) > 0.01:
            warnings.append(
                f"Weights add up to {self.total_weight():g}%, not 100%."
            )
        # Scores outside 0-100 are almost certainly typos.
        for item in self.items:
            if item.score is not None and not 0 <= item.score <= 100:
                warnings.append(f"{item.name}: score {item.score:g} is outside 0-100.")
        # Duplicate names would make pins and sliders ambiguous.
        seen: set[str] = set()
        for item in self.items:
            if item.name in seen:
                warnings.append(f"Duplicate item name: {item.name}.")
            seen.add(item.name)
        return warnings


def split_component(name: str, total_weight: float, count: int, kind: Kind) -> list[Item]:
    """Split one component into equal numbered items.

    Args:
        name: Base name, e.g. "Quiz".
        total_weight: Combined weight of all pieces, e.g. 30.
        count: How many pieces, e.g. 6.
        kind: Kind given to every piece.

    Returns:
        Items "Quiz 1" ... "Quiz 6" at 5% each.
    """
    return [
        Item(f"{name} {number}", total_weight / count, kind)
        for number in range(1, count + 1)
    ]


def _statistics_course() -> Course:
    """Build a fresh Statistics & Probability course with no scores."""
    items = [Item("Class & Tutorial Participation", 5.0, "other")]
    items += split_component("Quiz", 30.0, 6, "quiz")
    items += [
        Item("Programming Exercise", 5.0, "other"),
        Item("Mid Term Exam", 30.0, "exam"),
        Item("Final Exam", 30.0, "exam"),
    ]
    return Course("Statistics & Probability", items)


# Built-in course layouts. The app copies these, so edits never change them.
PRESETS: dict[str, Course] = {"Statistics & Probability": _statistics_course()}

# Example scores for the preset, keyed by item name. Items not listed stay empty.
DEMO_SCORES: dict[str, float] = {
    "Class & Tutorial Participation": 90,
    "Quiz 1": 88,
    "Quiz 2": 76,
    "Quiz 3": 82,
    "Programming Exercise": 85,
    "Mid Term Exam": 72,
}
