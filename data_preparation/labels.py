from typing import Dict, List, Tuple

from rapidfuzz import fuzz


VALID_FIELDS = {"house_number", "street", "city", "postal_code", "state", "country"}

MAX_TOKENS = {
    "city": 3,
    "street": 4,
    "state": 3,
}


def align_labels(raw_address: str, parsed: Dict[str, str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Align dict-style address fields to whitespace tokens using BIO tags."""
    tokens = raw_address.split()
    labels: List[str] = ["O"] * len(tokens)

    sorted_fields = sorted(
        [f for f in parsed if f in VALID_FIELDS and parsed[f]],
        key=lambda f: len(str(parsed[f])),
        reverse=True,
    )

    for field in sorted_fields:
        value_tokens = str(parsed[field]).split()
        window_size = min(len(value_tokens), MAX_TOKENS.get(field, len(value_tokens)))

        best_score, best_i = 0, -1

        for i in range(len(tokens) - window_size + 1):
            if any(labels[i + k] != "O" for k in range(window_size)):
                continue

            window = " ".join(tokens[i : i + window_size])
            candidate = " ".join(value_tokens[:window_size])

            if window.lower() == candidate.lower():
                best_score, best_i = 100, i
                break

            score = fuzz.ratio(window.lower(), candidate.lower())
            if score > best_score:
                best_score, best_i = score, i

        if best_score >= 60 and best_i >= 0:
            labels[best_i] = f"B-{field}"
            for k in range(1, window_size):
                labels[best_i + k] = f"I-{field}"

    return tuple(tokens), tuple(labels)


def bio_to_dict(tokens: Tuple[str, ...], labels: List[str]) -> Dict[str, str]:
    """Convert BIO sequence labels back into a dictionary of address fields."""
    result: Dict[str, str] = {}
    current_field = None
    current_tokens: List[str] = []

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_field:
                result[current_field] = " ".join(current_tokens)
            current_field = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and label[2:] == current_field:
            current_tokens.append(token)
        else:
            if current_field:
                result[current_field] = " ".join(current_tokens)
            current_field, current_tokens = None, []

    if current_field:
        result[current_field] = " ".join(current_tokens)

    return result
