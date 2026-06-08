from __future__ import annotations


def normalize_style_mode(style: str = "quality", fallback: str = "quality", look_safe: bool = True) -> str:
    """Return the single supported creative profile for the core route.

    Older branches accepted filename/style aliases such as cyber, warm, lowkey and
    cinematic. The current renderer keeps style expression in the continuous
    background-derived LookPolicy, so discrete style aliases are intentionally not
    routed here.
    """
    return "quality"
