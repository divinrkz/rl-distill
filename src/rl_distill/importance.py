"""Sentence importance via resampling, and aggregation into a category profile.

Resampling importance (accuracy), following Thought Anchors: resample rollouts
from each sentence boundary and measure how keeping a sentence shifts the
probability of a correct answer.

    acc[i]        = P(correct | resample from boundary i, sentence i onward regenerated)
    importance[i] = acc[i+1] - acc[i]   (effect of keeping sentence i)

The counterfactual semantic-dedup refinement (filtering rollouts whose resampled
sentence is dissimilar to the original) is deferred to the full run; this is the
base metric that gets the pipeline end to end.
"""

from collections import defaultdict


def select_boundaries(offsets, max_prefixes):
    """Evenly subsample sentence boundaries to at most max_prefixes.

    Bounds resampling cost on long CoTs (OpenThinker) while leaving short ones
    (ORZ) untouched. Importance then measures the effect of each ~block of
    sentences rather than each single sentence.
    """
    if not max_prefixes or len(offsets) <= max_prefixes:
        return offsets
    step = len(offsets) / max_prefixes
    idx = sorted({int(k * step) for k in range(max_prefixes)})
    return [offsets[i] for i in idx]


def build_prefixes(thinking, offsets):
    """One prefix per sentence boundary: thinking[:offset_i], plus full thinking.

    Returns len(offsets)+1 prefixes; prefix i excludes sentence i onward.
    """
    positions = [off for _, off in offsets]
    prefixes = [thinking[:p] for p in positions]
    prefixes.append(thinking)
    return prefixes


def accuracies(graded):
    """graded: per-prefix list of bools -> per-prefix accuracy."""
    return [sum(g) / len(g) if g else 0.0 for g in graded]


def sentence_importance(acc):
    """importance[i] = acc[i+1] - acc[i] for each sentence i."""
    return [acc[i + 1] - acc[i] for i in range(len(acc) - 1)]


def threshold_importance(importance, threshold):
    """Zero out importances below the noise floor (|imp| < threshold).

    At n rollouts, importance resolution is ~1/n, so sub-threshold values are
    indistinguishable from noise and should not accumulate in the profile.
    """
    return [i if abs(i) >= threshold else 0.0 for i in importance]


def aggregate_profiles(profiles, categories):
    """Mean and population-std of per-problem profiles, per category.

    Averaging per-problem profiles (each summing to 1) weights problems equally,
    which normalizes for CoT length.
    """
    import statistics

    mean, std = {}, {}
    for c in categories:
        vals = [p.get(c, 0.0) for p in profiles]
        mean[c] = sum(vals) / len(vals) if vals else 0.0
        std[c] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def category_profile(labels, importance, categories):
    """Distribution of importance magnitude over categories (sums to 1).

    Returns (profile, signed_by_category, counts).
    """
    mag = defaultdict(float)
    signed = defaultdict(float)
    counts = defaultdict(int)
    for label, imp in zip(labels, importance):
        mag[label] += abs(imp)
        signed[label] += imp
        counts[label] += 1

    total = sum(mag.values()) or 1.0
    profile = {c: mag.get(c, 0.0) / total for c in categories}
    signed_by_category = {c: signed.get(c, 0.0) for c in categories}
    counts_by_category = {c: counts.get(c, 0) for c in categories}
    return profile, signed_by_category, counts_by_category
