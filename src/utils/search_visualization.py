"""Plot accepted autonomous-search history."""
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt


def plot_search_history(events: Iterable[Dict[str, Any]], output_path: str | Path) -> Path:
    """Save accepted validation accuracy and physical parameter count by iteration."""
    accepted = []
    for event in events:
        candidates = event.get("candidates")
        if isinstance(candidates, list):
            accepted.extend(
                dict(candidate, iteration=event.get("iteration"))
                for candidate in candidates
                if candidate.get("final_action") == "accept" and "validation" in candidate
            )
        elif event.get("final_action") == "accept" and "validation" in event:
            accepted.append(event)
    if not accepted:
        raise ValueError("at least one accepted event is required to plot history")

    iterations = [event["iteration"] for event in accepted]
    accuracy = [event["validation"]["accuracy"] for event in accepted]
    parameters = [
        event.get("actual_parameter_count", event["cheap_critic"]["parameter_count"])
        for event in accepted
    ]

    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    axes[0].plot(iterations, accuracy, color="#0072B2", linewidth=2, marker="o", markersize=6)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axes[1].plot(iterations, parameters, color="#D55E00", linewidth=2, marker="o", markersize=6)
    axes[1].set_xlabel("Accepted iteration")
    axes[1].set_ylabel("Parameters")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
