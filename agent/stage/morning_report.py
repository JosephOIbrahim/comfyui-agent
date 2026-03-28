"""Morning Report Generator — summarizes autoresearch session results.

Generates a formatted markdown report from ratchet history, experience
statistics, and FORESIGHT prediction data. Designed for the artist to
review what happened overnight.
"""

from __future__ import annotations

from typing import Any


def generate_report(
    ratchet_history: list[dict[str, Any]] | None = None,
    experience_stats: dict[str, Any] | None = None,
    *,
    session_name: str = "Autoresearch",
    counterfactual_count: int = 0,
    warnings: list[str] | None = None,
    program_objective: str = "",
) -> str:
    """Generate a morning report as formatted markdown.

    Args:
        ratchet_history: List of RatchetDecision dicts from ratchet.history.
        experience_stats: Dict from experience.get_statistics().
        session_name: Name of the session for the report header.
        counterfactual_count: Number of counterfactuals generated.
        warnings: List of warning messages to include.
        program_objective: The program's objective statement.

    Returns:
        Formatted markdown string.
    """
    lines: list[str] = []
    history = ratchet_history or []
    stats = experience_stats or {}

    # Header
    lines.append(f"# {session_name} — Morning Report")
    lines.append("")

    if program_objective:
        lines.append(f"**Objective:** {program_objective}")
        lines.append("")

    # Experiment summary
    total = len(history)
    kept = sum(1 for d in history if d.get("kept", False))
    discarded = total - kept
    keep_rate = (kept / total * 100) if total > 0 else 0.0

    lines.append("## Experiment Summary")
    lines.append(f"- **Total experiments:** {total}")
    lines.append(f"- **Kept:** {kept} ({keep_rate:.0f}%)")
    lines.append(f"- **Discarded:** {discarded}")
    lines.append("")

    # Score trajectory
    if total > 0:
        composites = [d.get("composite", 0.0) for d in history]
        lines.append("## Score Trajectory")
        lines.append(f"- **First:** {composites[0]:.3f}")
        lines.append(f"- **Last:** {composites[-1]:.3f}")
        lines.append(f"- **Best:** {max(composites):.3f}")
        lines.append(f"- **Worst:** {min(composites):.3f}")
        lines.append(f"- **Average:** {sum(composites) / len(composites):.3f}")

        # Trend
        if len(composites) >= 3:
            first_third = sum(composites[:len(composites)//3]) / max(len(composites)//3, 1)
            last_third = sum(composites[-len(composites)//3:]) / max(len(composites)//3, 1)
            if last_third > first_third + 0.05:
                lines.append("- **Trend:** Improving")
            elif last_third < first_third - 0.05:
                lines.append("- **Trend:** Declining")
            else:
                lines.append("- **Trend:** Stable")
        lines.append("")

    # Most impactful axes
    if total > 0:
        lines.append("## Most Impactful Axes")
        axis_impact = _compute_axis_impact(history)
        for axis, impact in sorted(axis_impact.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- **{axis}:** impact score {impact:.3f}")
        lines.append("")

    # Best recipe parameters
    if total > 0:
        best = max(history, key=lambda d: d.get("composite", 0.0))
        lines.append("## Best Recipe")
        lines.append(f"- **Composite score:** {best.get('composite', 0.0):.3f}")
        lines.append(f"- **Delta ID:** {best.get('delta_id', '?')}")
        scores = best.get("axis_scores", {})
        if scores:
            for axis, score in sorted(scores.items()):
                lines.append(f"  - {axis}: {score:.3f}")
        lines.append("")

    # FORESIGHT prediction accuracy
    predicted = [d for d in history if d.get("prediction_accuracy") is not None]
    if predicted:
        accuracies = [d["prediction_accuracy"] for d in predicted]
        lines.append("## FORESIGHT Prediction Accuracy")
        lines.append(f"- **Predictions made:** {len(predicted)}")
        lines.append(f"- **Average accuracy:** {sum(accuracies) / len(accuracies):.3f}")
        lines.append(f"- **Best accuracy:** {max(accuracies):.3f}")
        lines.append(f"- **Worst accuracy:** {min(accuracies):.3f}")
        lines.append("")

    # Experience stats
    if stats.get("total_count", 0) > 0:
        lines.append("## Experience Base")
        lines.append(f"- **Total experiences:** {stats['total_count']}")
        lines.append(f"- **Unique signatures:** {stats.get('unique_signatures', 0)}")
        avg_outcome = stats.get("avg_outcome", {})
        if avg_outcome:
            lines.append("- **Average outcomes:**")
            for axis, score in sorted(avg_outcome.items()):
                lines.append(f"  - {axis}: {score:.3f}")
        lines.append("")

    # Counterfactuals
    if counterfactual_count > 0:
        lines.append("## Counterfactuals")
        lines.append(f"- **Generated:** {counterfactual_count}")
        lines.append("- These represent 'what if' alternatives for future validation.")
        lines.append("")

    # Warnings
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def _compute_axis_impact(history: list[dict]) -> dict[str, float]:
    """Compute impact score per axis based on variance in kept vs discarded.

    Higher impact = axis scores differ more between kept and discarded.
    """
    kept_scores: dict[str, list[float]] = {}
    disc_scores: dict[str, list[float]] = {}

    for d in history:
        scores = d.get("axis_scores", {})
        target = kept_scores if d.get("kept", False) else disc_scores
        for axis, score in scores.items():
            target.setdefault(axis, []).append(score)

    impact: dict[str, float] = {}
    all_axes = set(kept_scores) | set(disc_scores)

    for axis in all_axes:
        k = kept_scores.get(axis, [])
        d = disc_scores.get(axis, [])
        k_avg = sum(k) / len(k) if k else 0.0
        d_avg = sum(d) / len(d) if d else 0.0
        impact[axis] = abs(k_avg - d_avg)

    return impact
