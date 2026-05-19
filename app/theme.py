"""Grafana-inspired dark theme tokens (deep navy + warm accents)."""

PALETTE = {
    "bg_primary":   "#0d1117",   # main background (GitHub-dark / Grafana base)
    "bg_secondary": "#161b22",   # panel / sidebar
    "bg_card":      "#1c2128",   # stat blocks
    "border":       "#30363d",
    # Grafana-style status colors
    "ok":      "#3fb950",   # green
    "warn":    "#d29922",   # amber
    "alert":   "#f85149",   # red
    "info":    "#58a6ff",   # blue (no-status)
    "muted":   "#8b949e",   # secondary text
    "text":    "#e6edf3",   # primary text
    # Chart series palette (Grafana 7-color)
    "series_temp":     "#ff9830",
    "series_humidity": "#5794f2",
    "series_pressure": "#73bf69",
    "series_lux":      "#b877d9",
    "series_current":  "#ff7383",
    "series_forecast": "#73e0a9",
    # Backwards-compat aliases (used by older charts code)
    "accent_blue":  "#5794f2",
    "accent_amber": "#ff9830",
    "accent_red":   "#ff7383",
    "accent_green": "#73bf69",
}


def plotly_theme() -> dict:
    """Plotly layout patch matching the Grafana-style dark UI.

    NB: `uirevision='static'` ensures user pan/zoom survive figure replots -
    critical for the live charts page (otherwise every tick resets viewport).
    """
    return {
        "paper_bgcolor": PALETTE["bg_card"],
        "plot_bgcolor":  PALETTE["bg_card"],
        "font": {"color": PALETTE["text"], "size": 12, "family": "Inter, sans-serif"},
        "xaxis": {
            "showgrid": True,
            "gridcolor": PALETTE["border"],
            "zeroline": False,
            "color": PALETTE["text"],
            "tickfont": {"color": PALETTE["muted"]},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": PALETTE["border"],
            "zeroline": False,
            "color": PALETTE["text"],
            "tickfont": {"color": PALETTE["muted"]},
        },
        "legend": {
            "font": {"color": PALETTE["text"]},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
        },
        "margin": {"t": 40, "r": 20, "b": 40, "l": 50},
        "hovermode": "x unified",
        "uirevision": "static",          # preserves zoom/pan across replots
    }


def status_color(severity: str) -> str:
    """Map advisory severity -> palette colour."""
    return {
        "alert": PALETTE["alert"],
        "warn":  PALETTE["warn"],
        "info":  PALETTE["info"],
        "ok":    PALETTE["ok"],
    }.get(severity, PALETTE["muted"])
