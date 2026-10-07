"""Attack traffic layered onto the benign baseline.

An attack is built as its own packet frame and then interleaved into a baseline, so the
output of one pass is valid input to the next. Baseline rows are never rewritten, which
keeps their benign labels correct as ground truth when several attacks stack on one
capture.
"""

import pandas as pd

from peruze.capture import schema


def interleave(baseline: pd.DataFrame, attack: pd.DataFrame) -> pd.DataFrame:
    """Place attack rows into a baseline by arrival time, under the packet contract."""
    combined = pd.concat([baseline, attack], ignore_index=True)
    # Timestamps can collide, and a stable sort puts the baseline packet first at those ties instead of leaving the row order undefined.
    return schema.validate_frame(combined.sort_values("ts", kind="stable", ignore_index=True))
