'''Color palette definitions for visualization.'''

### IMPORTS ###

import seaborn as sns

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

###########################


sns.set(style="whitegrid")

MODEL_COLOR_SEQUENCE = ["#0173B2", "#DE8F05", "#029E73"]
DOMAIN_COLOR_SEQUENCE = ["#D55E00", "#CC78BC", "#CA9161"]
SHOT_COLOR_SEQUENCE = ["#949494", "#ECE133", "#56B4E9"]

COLOR_SEQUENCE = MODEL_COLOR_SEQUENCE + DOMAIN_COLOR_SEQUENCE + SHOT_COLOR_SEQUENCE

MODEL_ORDER = [
    "kimi-k2-instruct-0905",
    "llama-3.1-8b-instruct",
    "qwen3-coder-30b-a3b-instruct",
]
DOMAIN_ORDER = ["hatecrime", "employment", "lending"]
SHOT_ORDER = ["zero", "one", "few"]

MODEL_PALETTE = {
    model: MODEL_COLOR_SEQUENCE[i % len(MODEL_COLOR_SEQUENCE)]
    for i, model in enumerate(MODEL_ORDER)
}
DOMAIN_PALETTE = {
    domain: DOMAIN_COLOR_SEQUENCE[i % len(DOMAIN_COLOR_SEQUENCE)]
    for i, domain in enumerate(DOMAIN_ORDER)
}
SHOT_PALETTE = {
    shot: SHOT_COLOR_SEQUENCE[i % len(SHOT_COLOR_SEQUENCE)]
    for i, shot in enumerate(SHOT_ORDER)
}

MODEL_ALIASES = {
    "kimi-k2-instruct-0905": "kimi-k2-instruct-0905",
    "llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b-instruct",
    "qwen3-coder-30b-a3b-instruct": "qwen3-coder-30b-a3b-instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen3-coder-30b-a3b-instruct",
}


def ordered_levels(values, reference_order):
    observed = set(values)
    return [item for item in reference_order if item in observed]


def _fallback_color(label):
    return COLOR_SEQUENCE[hash(label) % len(COLOR_SEQUENCE)]


def model_color(label: str) -> str:
    canonical = MODEL_ALIASES.get(label, label)
    return MODEL_PALETTE.get(canonical, _fallback_color(canonical))


def model_color_map(labels):
    return {label: model_color(label) for label in labels}


def domain_color(label: str) -> str:
    canonical = label.lower()
    return DOMAIN_PALETTE.get(canonical, _fallback_color(canonical))


def shot_color(label: str) -> str:
    canonical = label.lower()
    return SHOT_PALETTE.get(canonical, _fallback_color(canonical))