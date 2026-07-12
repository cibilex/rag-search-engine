def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    return [vec1[i] + vec2[i] for i in range(len(vec1))]


def dot(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    total = 0.0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    return [vec1[i] - vec2[i] for i in range(len(vec1))]
