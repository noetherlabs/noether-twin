import random

import numpy as np
import pytest

SEED = 20260726


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
