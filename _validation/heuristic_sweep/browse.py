#!/usr/bin/env python3
"""Map browser for the TFM heuristic sweep -- a thin viewer over PRE-RENDERED cards.

The old browser read the displacement cache and re-solved traction on every
interaction. Now the GT-tuned oracle force maps are cached (build_force_cache.py)
and rendered once into comparison cards (render_cache.py), so this does no numerics
at all: it reads renders/index.csv and shows the matching PNG.

Navigation is a PARAMETER SET, like the GUI tools: you dial in the scenario
(imaging condition, jitter, dipole footprint / cell, peak displacement) and the
reconstruction (method, resolution, smoothing) with sliders, and the matching 3x3
card appears. The sweep's whole point -- which config recovers the most force -- is
kept as a "snap to best" button plus a rank readout, not a table to scroll.

Run (after render_cache.py has populated $STAGE/renders):
    python browse.py                  # -> http://127.0.0.1:8053
    STAGE=/home/aruppel/Data/tfm_heuristic python browse.py
"""
from __future__ import annotations
import os, csv, re
from collections import defaultdict

import flask
from dash import Dash, dcc, html, Input, Output, State, no_update

STAGE = os.environ.get("STAGE", "/home/aruppel/Data/tfm_heuristic")
RENDERS = os.path.join(STAGE, "renders")
INDEX = os.path.join(RENDERS, "index.csv")

INPUT_RE = re.compile(r"^(PIV|FFD|ILK)_res(\d+)_sm(\d+)$")
METHODS = ["PIV", "FFD", "ILK"]
KNOB_LABEL = {"PIV": "window", "FFD": "spacing", "ILK": "radius"}


# --------------------------------------------------------------------------- #
# Data: read the baked index once (cheap; re-read on process restart only).
# --------------------------------------------------------------------------- #
def load_index():
    """index.csv -> row dicts + lookup tables keyed for the slider UI.

    LOOK[(cond,scene,input)] = row           (one card)
    RANK[(cond,scene)]       = [rows sorted by force ceiling, best first]
    RESVAL[method][res_idx]  = actual knob value (px)   e.g. PIV res2 -> window 16
    """
    LOOK, by_scene, RESVAL = {}, defaultdict(list), defaultdict(dict)
    if not os.path.exists(INDEX):
        return LOOK, {}, {}, {}
    with open(INDEX) as fh:
        for r in csv.DictReader(fh):
            fttc, l1 = float(r["fttc_obj"]), float(r["l1_obj"])
            r["fttc_obj"], r["l1_obj"] = round(fttc, 3), round(l1, 3)
            r["disp_nrmse"] = round(float(r["disp_nrmse"]), 3)
            r["best"] = round(min(fttc, l1), 3)
            LOOK[(r["cond"], r["scene"], r["input"])] = r
            by_scene[(r["cond"], r["scene"])].append(r)
            m = INPUT_RE.match(r["input"])
            if m:
                RESVAL[m.group(1)][int(m.group(2))] = int(float(r["res_val"]))
    RANK = {k: sorted(v, key=lambda x: x["best"]) for k, v in by_scene.items()}
    return LOOK, RANK, dict(RESVAL), by_scene


LOOK, RANK, RESVAL, BY_SCENE = load_index()


def _uniq_sorted(vals):
    return sorted(set(vals), key=lambda s: float(s))


# Axis token sets, mined from the actual scene names (never reformatted).
_dip = [(c, s) for (c, s) in BY_SCENE if c.startswith("s") and "_j" in c]
S_LEVELS = sorted({int(re.match(r"s(\d+)_j\d+", c).group(1)) for c, _ in _dip})
J_LEVELS = sorted({int(re.search(r"_j(\d+)", c).group(1)) for c, _ in _dip})
F_TOKS = _uniq_sorted(re.match(r"f([^_]+)_u", s).group(1) for _, s in _dip)
U_TOKS = _uniq_sorted(re.search(r"_u(.+)$", s).group(1) for _, s in _dip)
_cell = [(c, s) for (c, s) in BY_SCENE if c.startswith("cell")]
CELL_COND = sorted({c for c, _ in _cell})
CELL_IDX = sorted({re.match(r"synth(\d+)_u", s).group(1) for _, s in _cell})
CELL_U = _uniq_sorted(re.search(r"_u(.+)$", s).group(1) for _, s in _cell)
RES_MAX = max((i for m in RESVAL.values() for i in m), default=0)
J_NAME = {1: "mild", 3: "severe"}


# --------------------------------------------------------------------------- #
# App + a static route so <img> can fetch the PNGs straight off disk.
# --------------------------------------------------------------------------- #
server = flask.Flask(__name__)
app = Dash(__name__, server=server)
app.title = "TFM sweep - card browser"


@server.route("/card/<path:relpath>")
def _card(relpath):
    return flask.send_from_directory(RENDERS, relpath)


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
_RAIL = {"width": "380px", "flex": "0 0 380px", "position": "sticky", "top": "10px",
         "alignSelf": "flex-start", "paddingRight": "18px",
         "borderRight": "1px solid #e4e3de", "maxHeight": "98vh", "overflowY": "auto"}
_LBL = {"fontSize": "11px", "fontWeight": 700, "color": "#0b0b0b",
        "margin": "14px 0 2px", "letterSpacing": ".02em"}
_SECT = {"fontSize": "10px", "fontWeight": 800, "color": "#9467bd",
         "textTransform": "uppercase", "letterSpacing": ".08em",
         "margin": "18px 0 2px", "borderBottom": "1px solid #eee", "paddingBottom": "2px"}


def _slider(cid, tokens, fmt=str, value=0):
    """Index slider (keys 0..n-1) whose marks show the real token value."""
    marks = {i: {"label": fmt(t), "style": {"fontSize": "9px"}} for i, t in enumerate(tokens)}
    return dcc.Slider(id=cid, min=0, max=len(tokens) - 1, step=None, value=value,
                      marks=marks, included=False)


def _radio(cid, options, value, inline=True):
    return dcc.RadioItems(
        id=cid, value=value,
        options=[{"label": lab, "value": val} for lab, val in options],
        inline=inline, inputStyle={"marginRight": "3px", "marginLeft": "9px"},
        style={"fontSize": "12px"})


app.layout = html.Div([
    html.Div([
        html.H3("TFM sweep card browser", style={"margin": "0 0 2px"}),
        html.Div("dial in a scenario + reconstruction · oracle FTTC+L2 vs FISTA+L1",
                 style={"fontSize": "10.5px", "color": "#52514e"}),

        html.Div("SCENARIO", style=_SECT),
        _radio("family", [("dipole", "dipole"), ("cell", "cell")], "dipole"),

        # --- dipole scenario controls ---
        html.Div([
            html.Div("imaging scenario  (bead density × NA/PSF × SNR)", style=_LBL),
            _slider("s_lvl", S_LEVELS, fmt=lambda s: f"s{s}"),
            html.Div("registration jitter", style=_LBL),
            _radio("jit", [(f"{J_NAME.get(j, j)} (j{j})", j) for j in J_LEVELS], J_LEVELS[0]),
            html.Div("dipole footprint  f", style=_LBL),
            _slider("foot", F_TOKS),
            html.Div("peak displacement  u  (px)", style=_LBL),
            _slider("u_dip", U_TOKS),
        ], id="dip_box"),

        # --- cell scenario controls ---
        html.Div([
            html.Div(f"synthetic cell  ({CELL_COND[0] if CELL_COND else '—'})", style=_LBL),
            _slider("cell_i", CELL_IDX, fmt=lambda i: f"#{i}"),
            html.Div("peak displacement  u  (px)", style=_LBL),
            _slider("u_cell", CELL_U),
        ], id="cell_box", style={"display": "none"}),

        html.Div("RECONSTRUCTION", style=_SECT),
        html.Div("method", style=_LBL),
        _radio("method", [(m, m) for m in METHODS], "PIV"),
        html.Div("resolution  (coarse → fine)", style=_LBL),
        dcc.Slider(id="res", min=0, max=RES_MAX, step=None, value=0, included=False,
                   marks={i: str(i) for i in range(RES_MAX + 1)}),
        html.Div("smoothing  (PIV only)", style=_LBL),
        _radio("smooth", [("off", 0), ("on", 1)], 0),

        html.Button("→ snap to best for this scene", id="snap",
                    style={"marginTop": "18px", "fontSize": "11px", "padding": "5px 10px",
                           "cursor": "pointer", "width": "100%"}),

        html.Div(id="caption", style={"fontSize": "11px", "fontFamily": "monospace",
                                      "marginTop": "14px", "color": "#0b0b0b",
                                      "lineHeight": "1.5", "whiteSpace": "pre-wrap"}),
    ], style=_RAIL),

    html.Div([html.Img(id="card", style={"width": "100%", "maxWidth": "1150px",
                                         "border": "1px solid #eee"})],
             style={"flex": "1 1 auto", "paddingLeft": "16px", "minWidth": "0"}),
], style={"display": "flex", "fontFamily": "system-ui, sans-serif", "padding": "12px 16px"})


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@app.callback(Output("dip_box", "style"), Output("cell_box", "style"),
              Input("family", "value"))
def _toggle_family(fam):
    show, hide = {}, {"display": "none"}
    return (show, hide) if fam == "dipole" else (hide, show)


@app.callback(Output("res", "marks"), Input("method", "value"))
def _res_marks(method):
    rv = RESVAL.get(method, {})
    return {i: {"label": str(rv[i]) if i in rv else "—",
                "style": {"fontSize": "9px", "color": "#0b0b0b" if i in rv else "#c9c7c1"}}
            for i in range(RES_MAX + 1)}


@app.callback(Output("card", "src"), Output("caption", "children"),
              Input("family", "value"),
              Input("s_lvl", "value"), Input("jit", "value"),
              Input("foot", "value"), Input("u_dip", "value"),
              Input("cell_i", "value"), Input("u_cell", "value"),
              Input("method", "value"), Input("res", "value"), Input("smooth", "value"))
def _show(fam, s, jit, foot, u_dip, cell_i, u_cell, method, res, smooth):
    if fam == "dipole":
        cond, scene = f"s{S_LEVELS[s]}_j{jit}", f"f{F_TOKS[foot]}_u{U_TOKS[u_dip]}"
    else:
        cond, scene = CELL_COND[0], f"synth{CELL_IDX[cell_i]}_u{CELL_U[u_cell]}"

    # smoothing is a PIV-only axis; resolution is clamped to what the method has.
    sm = smooth if method == "PIV" else 0
    avail = sorted(RESVAL.get(method, {}))
    res_used = min(res, avail[-1]) if avail else res       # clamp to the method's finest
    inp = f"{method}_res{res_used}_sm{sm}"
    row = LOOK.get((cond, scene, inp))
    if not row:
        return no_update, f"no card for {cond}/{scene}  ·  {inp}"

    knob = RESVAL.get(method, {}).get(res_used)
    ranking = RANK.get((cond, scene), [])
    rank_i = next((k for k, rr in enumerate(ranking) if rr["input"] == inp), None)
    best = ranking[0] if ranking else None

    lo = "nRMSE" if row["objective"] == "nrmse" else "Sabass J"
    clamp = f"   (clamped: {method} max res {avail[-1]})" if res_used != res else ""
    smtxt = ("on" if sm else "off") if method == "PIV" else "n/a (PIV only)"
    cap = "\n".join([
        f"{cond} / {scene}   [{row['kind']}]",
        f"{method}  {KNOB_LABEL[method]}={knob}px  ·  res {res_used}{clamp}  ·  smoothing {smtxt}",
        "",
        f"disp nRMSE {row['disp_nrmse']}",
        f"FTTC+L2 {lo} {row['fttc_obj']}   FISTA+L1 {lo} {row['l1_obj']}   → winner {row['winner']}",
        "",
        (f"rank #{rank_i + 1} of {len(ranking)} by force ceiling"
         if rank_i is not None else f"({len(ranking)} configs for this scene)"),
        (f"best here: {best['input']}  ({lo} {best['best']}, {best['winner']})" if best else ""),
    ])
    return "/card/" + row["png"], cap


@app.callback(Output("method", "value"), Output("res", "value"), Output("smooth", "value"),
              Input("snap", "n_clicks"),
              State("family", "value"),
              State("s_lvl", "value"), State("jit", "value"),
              State("foot", "value"), State("u_dip", "value"),
              State("cell_i", "value"), State("u_cell", "value"),
              prevent_initial_call=True)
def _snap(_n, fam, s, jit, foot, u_dip, cell_i, u_cell):
    if fam == "dipole":
        cond, scene = f"s{S_LEVELS[s]}_j{jit}", f"f{F_TOKS[foot]}_u{U_TOKS[u_dip]}"
    else:
        cond, scene = CELL_COND[0], f"synth{CELL_IDX[cell_i]}_u{CELL_U[u_cell]}"
    ranking = RANK.get((cond, scene), [])
    if not ranking:
        return no_update, no_update, no_update
    m = INPUT_RE.match(ranking[0]["input"])
    if not m:
        return no_update, no_update, no_update
    return m.group(1), int(m.group(2)), int(m.group(3))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8053"))
    print(f"card browser -> http://127.0.0.1:{port}   (STAGE={STAGE}, {len(LOOK)} cards indexed)")
    app.run(debug=False, port=port)
