cat >/tmp/make_hog_synthetic_11.py <<'PY'
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Hog EQ synthetic BC dataset
#
# Policy identities MUST match CardDB.policy_identities():
#
# 0  hog_rider
# 1  firecracker
# 2  firecracker_evo
# 3  mighty_miner
# 4  mighty_miner_ability
# 5  tesla
# 6  tesla_evo
# 7  the_log
# 8  earthquake
# 9  skeletons
# 10 ice_spirit
# ------------------------------------------------------------

N = 20_000
rng = np.random.default_rng(42)

OUT = Path("data/synthetic_11/session_01")
OUT.mkdir(parents=True, exist_ok=True)

CARD_NAMES = [
    "hog_rider",
    "firecracker",
    "firecracker_evo",
    "mighty_miner",
    "mighty_miner_ability",
    "tesla",
    "tesla_evo",
    "the_log",
    "earthquake",
    "skeletons",
    "ice_spirit",
]

N_CARDS = len(CARD_NAMES)

# Action grid used by --size 432 = 18 x 24.
GRID_W = 18
GRID_H = 24
N_CELLS = GRID_W * GRID_H

# Observation expected by the current BC loader/model.
obs = rng.integers(
    0, 256,
    size=(N, 96, 64, 3),
    dtype=np.uint8,
)

# ------------------------------------------------------------
# Hand / next-card vectors
# ------------------------------------------------------------

hands = np.zeros((N, N_CARDS), dtype=np.float32)
nexts = np.zeros((N, N_CARDS), dtype=np.float32)

# ------------------------------------------------------------
# Elixir
# ------------------------------------------------------------

elixirs = rng.uniform(3.0, 10.0, size=(N, 1)).astype(np.float32)

# ------------------------------------------------------------
# Threat vector
#
# Current wide observation expects 52 values:
#
# 16 base threat
# 18 card identity/opponent memory
# 12 interactions
#  6 tower HP
# ------------------------------------------------------------

threats = rng.random((N, 52), dtype=np.float32)

# ------------------------------------------------------------
# Actions
#
# Expected format:
#   [wait/play, card_id, gx, gy]
#
# play = 1
# card_id = 0..10
# gx = 0..17
# gy = 0..23
# ------------------------------------------------------------

acts = np.zeros((N, 4), dtype=np.float32)

# Useful deployment regions.
#
# Own side is roughly y >= 12.
# Enemy side is roughly y < 12.
#
# Hog / Earthquake can target deeper areas.
# Defensive cards stay mostly on our side.
def random_cell(low_y, high_y):
    gx = rng.integers(0, GRID_W)
    gy = rng.integers(low_y, high_y + 1)
    return gx, gy


for i in range(N):

    # --------------------------------------------------------
    # Choose a plausible card.
    #
    # Bias toward actual gameplay decisions.
    # --------------------------------------------------------
    r = rng.random()

    if r < 0.25:
        card = 0                  # Hog Rider
    elif r < 0.34:
        card = 8                  # Earthquake
    elif r < 0.45:
        card = 5                  # Tesla
    elif r < 0.52:
        card = 6                  # Tesla Evo
    elif r < 0.61:
        card = 1                  # Firecracker
    elif r < 0.67:
        card = 2                  # Firecracker Evo
    elif r < 0.75:
        card = 3                  # Mighty Miner
    elif r < 0.79:
        card = 4                  # Mighty Miner ability
    elif r < 0.86:
        card = 7                  # The Log
    elif r < 0.94:
        card = 9                  # Skeletons
    else:
        card = 10                 # Ice Spirit

    # --------------------------------------------------------
    # Put the selected card into the hand.
    # --------------------------------------------------------
    hands[i, card] = 1.0

    # A random different next card.
    remaining = [x for x in range(N_CARDS) if x != card]
    next_card = rng.choice(remaining)
    nexts[i, next_card] = 1.0

    # --------------------------------------------------------
    # Synthetic action policy.
    # --------------------------------------------------------

    # Wait/play.
    acts[i, 0] = 1.0

    # Card.
    acts[i, 1] = float(card)

    if card == 0:
        # Hog Rider:
        # attack opponent tower / deep enemy-side placement.
        gx, gy = random_cell(3, 9)

    elif card == 8:
        # Earthquake:
        # buildings / tower region.
        gx, gy = random_cell(4, 10)

    elif card in (5, 6):
        # Tesla:
        # central defensive placement.
        gx = rng.integers(6, 13)
        gy = rng.integers(12, 19)

    elif card in (1, 2):
        # Firecracker:
        # defensive/support position.
        gx = rng.integers(4, 15)
        gy = rng.integers(14, 22)

    elif card == 3:
        # Mighty Miner.
        gx = rng.integers(4, 15)
        gy = rng.integers(12, 21)

    elif card == 4:
        # Mighty Miner ability has no placement.
        # Cell is ignored by the controller, but keep it legal.
        gx = 9
        gy = 18

    elif card == 7:
        # Log.
        gx = rng.integers(2, 16)
        gy = rng.integers(8, 16)

    elif card == 9:
        # Skeletons.
        gx = rng.integers(4, 15)
        gy = rng.integers(14, 22)

    else:
        # Ice Spirit.
        gx = rng.integers(4, 15)
        gy = rng.integers(13, 21)

    acts[i, 2] = float(gx)
    acts[i, 3] = float(gy)


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

np.savez_compressed(
    OUT / "dataset.npz",
    obs=obs,
    acts=acts,
    hands=hands,
    nexts=nexts,
    elixirs=elixirs,
    threats=threats,
    grid=np.array([GRID_W, GRID_H], dtype=np.int32),
    deck=np.array(CARD_NAMES),
)

print("Created:", OUT / "dataset.npz")
print("samples:", N)
print("obs:    ", obs.shape, obs.dtype)
print("acts:   ", acts.shape, acts.dtype)
print("hands:  ", hands.shape, hands.dtype)
print("nexts:  ", nexts.shape, nexts.dtype)
print("elixirs:", elixirs.shape, elixirs.dtype)
print("threats:", threats.shape, threats.dtype)
print("grid:   ", [GRID_W, GRID_H])
print("deck:   ", CARD_NAMES)
PY

python /tmp/make_hog_synthetic_11.py
