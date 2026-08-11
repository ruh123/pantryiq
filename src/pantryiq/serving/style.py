"""The page's visual layer: palette, chrome suppression, and the two figure forms.

Kept out of `app.py` so the app file stays about what is shown rather than how it looks.

**The colours are quoted, not chosen.** Surfaces, ink and status steps come from the reference
palette used elsewhere in this project's design work, whose contrasts are recorded against the
same light surface this page renders on (`#fcfcfb`): critical 4.68, good 3.27, warning 1.79.
Warning is sub-3:1 by design, which is why every status here ships an icon and a sentence — a
colour never carries the meaning on its own.

**Two figure forms, chosen by what the number is.**

A *stat tile* for the headline metrics: four of them is a KPI row, which is the right form for a
handful of standalone figures — a bar chart of four unrelated percentages would invite comparisons
between quantities that share no scale.

A *meter* for a ratio against a limit, and only where a limit actually exists. Nutrition coverage
has one (1.0 is complete, and 0.8 is the floor retrieval will answer from), so its fill carries
severity. `data_trust_score` has no published threshold, so its meter is a plain sequential blue:
inventing red/amber bands for it would be asserting a judgement the warehouse never made.
"""
from __future__ import annotations

# Reference palette — light surface.
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "rgba(11,11,11,0.10)"
BLUE = "#2a78d6"          # sequential step 450
BLUE_TRACK = "#cde2fb"    # step 100 — the unfilled track is a lighter step of the same ramp
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"

CSS = f"""
<style>
  /* Streamlit's own chrome — a demo URL should not offer to deploy itself. */
  [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{ display: none; }}
  [data-testid="stHeader"] {{ height: 0; background: transparent; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

  html, body, [data-testid="stAppViewContainer"] {{ background: {PLANE}; }}
  [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {HAIRLINE}; }}

  h1 {{ font-size: 1.9rem; font-weight: 640; letter-spacing: -0.018em; margin-bottom: .1rem; }}
  h4 {{ font-size: .82rem; font-weight: 620; letter-spacing: .07em; text-transform: uppercase;
       color: {MUTED}; margin: 2rem 0 .7rem; }}

  /* --- stat tile ------------------------------------------------------------------------ */
  .kpi-row {{ display: flex; gap: .7rem; margin: 1.1rem 0 .4rem; flex-wrap: wrap; }}
  .kpi {{ flex: 1 1 0; min-width: 190px; background: {SURFACE}; border: 1px solid {HAIRLINE};
          border-radius: 10px; padding: .85rem .95rem .9rem; }}
  .kpi-label {{ font-size: .74rem; color: {MUTED}; letter-spacing: .02em; margin-bottom: .3rem; }}
  /* Proportional figures: tabular-nums gives every digit a zero's width, which reads loose at
     display sizes. Reserved for columns that must align vertically. */
  .kpi-value {{ font-size: 1.85rem; font-weight: 640; color: {INK}; line-height: 1.05;
                letter-spacing: -0.02em; }}
  .kpi-note {{ font-size: .72rem; color: {INK_2}; margin-top: .3rem; line-height: 1.35; }}

  /* --- meter ---------------------------------------------------------------------------- */
  .meter {{ margin: .1rem 0 .55rem; }}
  .meter-head {{ display: flex; justify-content: space-between; align-items: baseline;
                 font-size: .72rem; color: {MUTED}; margin-bottom: .22rem; }}
  .meter-val {{ color: {INK}; font-weight: 620; font-variant-numeric: tabular-nums; }}
  .meter-track {{ height: 6px; border-radius: 3px; overflow: hidden; }}
  .meter-fill {{ height: 100%; border-radius: 3px; }}

  /* --- recipe card ---------------------------------------------------------------------- */
  .rc {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 10px;
         padding: .8rem .95rem .55rem; margin-bottom: .55rem; }}
  .rc-title {{ font-weight: 620; font-size: .96rem; color: {INK}; margin-bottom: .12rem; }}
  .rc-sub {{ font-size: .75rem; color: {MUTED}; margin-bottom: .5rem; }}
  .rc-fact {{ font-size: .82rem; color: {INK_2}; line-height: 1.5; }}
  .rc-fact b {{ color: {INK}; font-weight: 600; }}
  .rc-tag {{ display: inline-block; font-size: .68rem; color: {INK_2}; background: {PLANE};
             border: 1px solid {HAIRLINE}; border-radius: 999px; padding: .05rem .45rem;
             margin-right: .25rem; }}

  /* --- stage strip ---------------------------------------------------------------------- */
  .stages {{ display: flex; gap: .55rem; margin: .4rem 0 0; flex-wrap: wrap; }}
  .stage {{ flex: 1 1 0; min-width: 175px; background: {SURFACE}; border: 1px solid {HAIRLINE};
            border-radius: 10px; padding: .7rem .85rem .75rem; position: relative; }}
  .stage-n {{ font-size: .68rem; color: {MUTED}; letter-spacing: .06em; }}
  .stage-name {{ font-size: .95rem; font-weight: 620; color: {INK}; margin: .15rem 0 .3rem; }}
  .stage-who {{ display: inline-block; font-size: .67rem; font-weight: 600; border-radius: 999px;
                padding: .08rem .45rem; margin-bottom: .4rem; }}
  .who-model {{ color: #7a4a10; background: rgba(250,178,25,.16); }}
  .who-code {{ color: #17466f; background: rgba(42,120,214,.12); }}
  .stage-what {{ font-size: .74rem; color: {INK_2}; line-height: 1.45; }}

  /* Ledger rows: Streamlit's default block gap plus a divider left ~100px of dead air per
     recipe, so eight of them scrolled forever. */
  .block-container hr {{ margin: .1rem 0 .75rem; border-color: {HAIRLINE}; }}
  [data-testid="stHorizontalBlock"] {{ margin-bottom: -.4rem; }}

  /* --- method steps --------------------------------------------------------------------- */
  .step {{ display: flex; gap: .6rem; align-items: baseline; font-size: .84rem; color: {INK_2};
           line-height: 1.55; margin-bottom: .3rem; }}
  .step-n {{ flex: 0 0 1.35rem; height: 1.35rem; border-radius: 50%; background: {PLANE};
             border: 1px solid {HAIRLINE}; color: {MUTED}; font-size: .68rem; font-weight: 620;
             text-align: center; line-height: 1.3rem; }}

  /* --- misc ----------------------------------------------------------------------------- */
  .answer {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 10px;
             padding: .3rem 1.1rem .5rem; margin-bottom: .7rem; }}
  .stamp {{ font-size: .72rem; color: {MUTED}; line-height: 1.5; }}
  .stamp code {{ font-size: .68rem; color: {INK_2}; }}
  div[data-testid="stAlert"] {{ border-radius: 9px; }}
  .stButton button {{ border-radius: 8px; border-color: {HAIRLINE}; font-size: .82rem; }}
</style>
"""


def stat_tile(label: str, value: str, note: str) -> str:
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>')


def meter(label: str, value: float, severity: bool = False) -> str:
    """A 0-1 ratio as a track and a fill.

    `severity=True` colours the fill by band and is used only for coverage, where the bands are
    the warehouse's own: 1.0 is complete, 0.8 is the floor retrieval will answer from at all.
    Everything else gets the sequential blue, because a number with no threshold has no severity.
    """
    if severity:
        fill = GOOD if value >= 1.0 else WARNING if value >= 0.8 else SERIOUS
        track = "rgba(11,11,11,0.07)"
    else:
        fill, track = BLUE, BLUE_TRACK
    width = max(0.0, min(1.0, value)) * 100
    return (f'<div class="meter"><div class="meter-head"><span>{label}</span>'
            f'<span class="meter-val">{value:.2f}</span></div>'
            f'<div class="meter-track" style="background:{track}">'
            f'<div class="meter-fill" style="width:{width:.1f}%;background:{fill}"></div>'
            f'</div></div>')
