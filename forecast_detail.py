"""
Bottom-up 2027 forecast detail: per service (AWS, GCP) and per GCP project.

Same basis as cloud_forecast.py (calendar-2026 actuals only), but every line gets its own model,
chosen by backtest — "use the method that would have best predicted this line's own recent months".

Candidate models (all start from the series itself; nothing is fitted across lines):
  Flat          last-3-month average held constant
  Damped trend  last-3-month average + Theil-Sen slope, momentum fading 15%/month (same as top-down Base)
  Linear        last-3-month average + Theil-Sen slope, undamped
  Holt (ETS)    statsmodels additive damped-trend exponential smoothing, parameters fitted to the line

Model selection: rolling-origin backtest inside 2026 — train on the first 5, 6, 7, 8 months, forecast up to
3 months ahead, score by mean absolute error (MAE). Ties within 5% go to the simpler model.
Range: approximate 80% interval = forecast ± 1.28 × backtest RMSE, widening with √horizon.
Guardrails: growth capped at 2× the exit run-rate; declines floored at 50% of it (both flagged); never below $0.
"All other" buckets always use the damped trend (they keep gaining new small lines, so selection would overreact).
Confidence: High (backtest error < 10% of run-rate), Medium (< 25%), Low (otherwise).

Audience split:
  * Leadership (CIO/CTO): leadership_section() — top services table + bottom-up vs top-down reconciliation.
  * SVP/VP: build_detail_report() — per-line charts, model evidence, GCP projects, families, prod vs non-prod,
    ramping projects and (optionally) an owner roll-up from gcp_project_owners.csv.
"""
from __future__ import annotations

import html
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cloud_forecast as cf
import finops_exec_report as fr
import multicloud_kpis as mk
from finops_exec_report import _money, _layout

# statsmodels ETS emits log(0) RuntimeWarnings on short, perfectly-fitted series; they don't affect the forecast.
warnings.filterwarnings('ignore', category=RuntimeWarning, module=r'statsmodels\.')

TOP_N = 15
MODELS = ['Flat', 'Damped trend', 'Linear', 'Holt (ETS)']          # simplest first (tie-break order)
TIE_TOLERANCE = 0.05
GROWTH_CAP = 2.0
DECLINE_FLOOR = 0.5
OWNERS_FILE = os.path.join(mk.HERE, 'gcp_project_owners.csv')
ENV_TOKENS = {'prod': 'prod', 'prd': 'prod', 'production': 'prod', 'dev': 'non-prod', 'nonprod': 'non-prod',
              'qa': 'non-prod', 'stg': 'non-prod', 'stage': 'non-prod', 'staging': 'non-prod', 'test': 'non-prod',
              'uat': 'non-prod', 'sandbox': 'non-prod', 'poc': 'non-prod'}


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def _predict(model: str, y: np.ndarray, h: int) -> np.ndarray:
    """Forecast h steps after the end of y."""
    steps = np.arange(1, h + 1)
    anchor = y[-3:].mean() if len(y) >= 3 else y.mean()
    if model == 'Flat':
        return np.full(h, anchor)
    slope = cf.theil_sen(pd.Series(y))
    if model == 'Damped trend':
        return anchor + slope * np.cumsum(cf.DAMPING ** steps)
    if model == 'Linear':
        return anchor + slope * steps
    if model == 'Holt (ETS)':
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            fit = ExponentialSmoothing(y.astype(float), trend='add', damped_trend=True,
                                       initialization_method='estimated').fit()
        out = np.asarray(fit.forecast(h), dtype=float)
        if not np.isfinite(out).all():
            raise ValueError('non-finite ETS forecast')
        return out
    raise ValueError(model)


def backtest(y: np.ndarray, model: str, min_train: int = 5, max_h: int = 3) -> np.ndarray:
    """Errors (forecast - actual) for every origin / horizon inside the series."""
    errs = []
    for t in range(min_train, len(y)):
        h = min(max_h, len(y) - t)
        pred = _predict(model, y[:t], h)
        errs.extend(pred - y[t:t + h])
    return np.array(errs)


@dataclass
class LineForecast:
    name: str
    actual: pd.Series
    forecast: pd.Series
    lo: pd.Series
    hi: pd.Series
    model: str
    mae: float
    mape: float
    scores: dict
    capped: bool = False
    floored: bool = False

    @property
    def confidence(self) -> str:
        e = self.mape
        return 'Low' if not np.isfinite(e) or e >= 0.25 else ('Medium' if e >= 0.10 else 'High')


def forecast_line(name: str, actual: pd.Series, horizon: pd.DatetimeIndex, force_model: str | None = None) -> LineForecast:
    y = actual.to_numpy(dtype=float)
    scores = {}
    for m in MODELS:
        try:
            e = backtest(y, m)
            scores[m] = dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))))
        except Exception:
            continue
    best_mae = min(v['mae'] for v in scores.values())
    model = force_model or next(m for m in MODELS if m in scores and scores[m]['mae'] <= best_mae * (1 + TIE_TOLERANCE))
    pred = _predict(model, y, len(horizon))
    anchor = y[-3:].mean()
    capped = floored = False
    if anchor > 0 and pred.max() > GROWTH_CAP * anchor:
        pred = np.minimum(pred, GROWTH_CAP * anchor)
        capped = True
    if anchor > 0 and pred.min() < DECLINE_FLOOR * anchor:
        pred = np.maximum(pred, DECLINE_FLOOR * anchor)
        floored = True
    pred = np.clip(pred, 0, None)
    rmse = scores[model]['rmse']
    width = 1.28 * rmse * np.sqrt(np.arange(1, len(horizon) + 1))
    recent = y[-3:].mean()
    return LineForecast(name, actual, pd.Series(pred, horizon), pd.Series(np.clip(pred - width, 0, None), horizon),
                        pd.Series(pred + width, horizon), model, scores[model]['mae'],
                        scores[model]['mae'] / recent if recent else np.nan, scores, capped, floored)


# --------------------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------------------
def project_family(name: str) -> tuple[str, str]:
    """'prj-edp-prod-edw' -> ('prj-edp-edw', 'prod'); environment tokens are stripped from the name."""
    if not isinstance(name, str) or name.startswith('('):
        return (name, 'shared')
    tokens = re.split(r'[-_]', name.lower())
    env = next((ENV_TOKENS[t] for t in tokens if t in ENV_TOKENS), 'unspecified')
    fam = '-'.join(t for t in tokens if t not in ENV_TOKENS) or name
    return fam, env


def load_owners() -> pd.DataFrame | None:
    if not os.path.exists(OWNERS_FILE):
        return None
    o = pd.read_csv(OWNERS_FILE, dtype=str).fillna('')
    for c in o.columns:
        o[c] = o[c].str.strip()
    o = o.drop_duplicates('project', keep='last')
    return o if 'owner' in o.columns and o['owner'].ne('').any() else None


SHARED_OWNER = 'Shared (no project)'      # GCP Support and other billing-account-level charges
UNASSIGNED = 'Unassigned'


def owner_map(owners: pd.DataFrame, projects) -> pd.Series:
    m = owners.set_index('project')['owner'].reindex(projects).fillna('')
    m[m.eq('') & m.index.str.startswith('(')] = SHARED_OWNER
    return m.replace('', UNASSIGNED)


def write_owner_template(projects: pd.Series) -> str:
    """Create gcp_project_owners.csv (project, family, environment, owner, department) for VPs to fill in."""
    if os.path.exists(OWNERS_FILE):
        return OWNERS_FILE
    rows = [dict(project=p, family=project_family(p)[0], environment=project_family(p)[1], owner='', department='')
            for p in projects]
    pd.DataFrame(rows).to_csv(OWNERS_FILE, index=False)
    return OWNERS_FILE


# --------------------------------------------------------------------------------------
# Detail forecast
# --------------------------------------------------------------------------------------
@dataclass
class Detail:
    fc: cf.Forecast
    lines: dict = field(default_factory=dict)        # key -> list[LineForecast] (top N + "All other")
    tables: dict = field(default_factory=dict)       # key -> summary DataFrame
    totals: dict = field(default_factory=dict)       # key -> bottom-up FY total
    model_check: pd.DataFrame | None = None          # backtest of the top-down totals
    families: pd.DataFrame | None = None
    env_split: pd.DataFrame | None = None
    ramping: pd.DataFrame | None = None
    owners: pd.DataFrame | None = None
    owner_projects: dict = field(default_factory=dict)  # owner -> per-project forecast table
    unassigned: pd.DataFrame | None = None


def _series_by(df, provider, dim, months_idx):
    d = df[(df['Provider'] == provider) & (df['Category'] == mk.SVC) & (df['Date'] >= months_idx[0]) & (df['Date'] <= months_idx[-1])]
    return d.pivot_table(index='Date', columns=dim, values='Cost', aggfunc='sum').reindex(months_idx).fillna(0)


def _summarize_lines(lines: list[LineForecast], fc: cf.Forecast) -> pd.DataFrame:
    yr = fc.months
    rows = []
    for lf in lines:
        exit_rr = lf.actual.tail(3).mean()
        fy = lf.forecast.loc[yr].sum()
        conf = f'{lf.confidence} (±{lf.mape:.0%})' if np.isfinite(lf.mape) else lf.confidence
        rows.append({'Line': lf.name, '2026 Jan–Sep': lf.actual.sum(), 'Exit run-rate /mo': exit_rr,
                     'Dec 2027 /mo': lf.forecast.iloc[-1], f'FY{cf.FORECAST_YEAR} forecast': fy,
                     'FY range low': lf.lo.loc[yr].sum(), 'FY range high': lf.hi.loc[yr].sum(),
                     'Δ vs exit × 12': fy - exit_rr * 12,
                     'Model': lf.model + (' (capped)' if lf.capped else '') + (' (floored)' if lf.floored else ''),
                     'Backtest error': lf.mape, 'Confidence': conf})
    return pd.DataFrame(rows).set_index('Line')


def build(df: pd.DataFrame, fc: cf.Forecast, top_n: int = TOP_N) -> Detail:
    dt = Detail(fc)
    months_idx = pd.date_range(fc.basis_start, fc.basis_end, freq='MS')
    specs = [('AWS services', mk.AWS, 'Service'), ('GCP services', mk.GCP, 'Service'), ('GCP projects', mk.GCP, 'project')]
    for key, p, dim in specs:
        s = _series_by(df, p, dim, months_idx)
        if p == mk.AWS and dim == 'Service':
            # same Savings Plans normalization as the top-down forecast
            gap = fc.normalization.get(mk.AWS)
            if gap is not None and len(gap) and fr.SP_NAME in s:
                s[fr.SP_NAME] = s[fr.SP_NAME] + gap.reindex(months_idx).fillna(0)
        order = s.sum().sort_values(ascending=False).index
        top, rest = order[:top_n], order[top_n:]
        lines = [forecast_line(n, s[n], fc.horizon) for n in top]
        lines.append(forecast_line(f'All other ({len(rest)})', s[rest].sum(axis=1), fc.horizon, force_model='Damped trend'))
        dt.lines[key] = lines
        dt.tables[key] = _summarize_lines(lines, fc)
        dt.totals[key] = dt.tables[key][f'FY{cf.FORECAST_YEAR} forecast'].sum()

    # Top-down model check: how would each method have done on the cloud totals?
    rows = []
    for p in (mk.AWS, mk.GCP):
        y = fc.svc_actual[p].to_numpy(dtype=float)
        if p == mk.AWS and fc.normalization.get(mk.AWS) is not None:
            y = y + fc.normalization[mk.AWS].reindex(fc.svc_actual.index).fillna(0).to_numpy()
        for m in MODELS:
            try:
                e = backtest(y, m)
                rows.append({'Cloud': p, 'Model': m, 'MAE /mo': np.mean(np.abs(e)), 'Error %': np.mean(np.abs(e)) / y[-3:].mean()})
            except Exception:
                pass
    dt.model_check = pd.DataFrame(rows)

    # GCP project families and environment split
    g = _series_by(df, mk.GCP, 'project', months_idx)
    fam = pd.DataFrame([dict(project=c, family=project_family(c)[0], env=project_family(c)[1]) for c in g.columns]).set_index('project')
    proj_lines = {lf.name: lf for lf in dt.lines['GCP projects']}
    # forecast families from family-level series (more stable than summing tiny projects)
    fs = g.T.groupby(fam['family']).sum().T
    fam_order = fs.sum().sort_values(ascending=False).index[:top_n]
    fam_lines = [forecast_line(n, fs[n], fc.horizon) for n in fam_order]
    fam_lines.append(forecast_line(f'All other families ({fs.shape[1] - len(fam_order)})', fs.drop(columns=fam_order).sum(axis=1),
                                   fc.horizon, force_model='Damped trend'))
    dt.lines['GCP project families'] = fam_lines
    t = _summarize_lines(fam_lines, fc)
    t.insert(0, 'Projects', [int((fam['family'] == n).sum()) if n in set(fam['family']) else int((~fam['family'].isin(fam_order)).sum()) for n in t.index])
    dt.tables['GCP project families'] = t
    dt.totals['GCP project families'] = t[f'FY{cf.FORECAST_YEAR} forecast'].sum()

    es = g.T.groupby(fam['env']).sum().T
    env_rows = []
    for env in es.columns:
        lf = forecast_line(env, es[env], fc.horizon)
        env_rows.append({'Environment': env, '2026 Jan–Sep': es[env].sum(), 'Exit run-rate /mo': es[env].tail(3).mean(),
                         f'FY{cf.FORECAST_YEAR} forecast': lf.forecast.loc[fc.months].sum(), 'Model': lf.model})
    env_df = pd.DataFrame(env_rows).set_index('Environment').sort_values('2026 Jan–Sep', ascending=False)
    env_df['Share of 2026'] = env_df['2026 Jan–Sep'] / env_df['2026 Jan–Sep'].sum()
    dt.env_split = env_df

    # Ramping / new projects: little or no spend in Q1 2026, material spend now
    q1 = g.iloc[:3].mean()
    exit_rr = g.tail(3).mean()
    ramp = pd.DataFrame({'Q1 2026 avg /mo': q1, 'Exit run-rate /mo': exit_rr})
    ramp = ramp[(ramp['Exit run-rate /mo'] >= 3_000) & (ramp['Exit run-rate /mo'] >= 2 * ramp['Q1 2026 avg /mo'].clip(lower=1))]
    ramp['First month billed'] = [g[c][g[c] > 100].index.min() for c in ramp.index]
    ramp[f'FY{cf.FORECAST_YEAR} forecast'] = [proj_lines[c].forecast.loc[fc.months].sum() if c in proj_lines
                                             else forecast_line(c, g[c], fc.horizon).forecast.loc[fc.months].sum() for c in ramp.index]
    dt.ramping = ramp.sort_values('Exit run-rate /mo', ascending=False)

    # Owner roll-up (optional mapping file)
    owners = load_owners()
    if owners is not None:
        _owner_rollup(dt, g, owners)
    return dt


def _owner_rollup(dt: Detail, g: pd.DataFrame, owners: pd.DataFrame) -> None:
    """Per-owner forecast = sum of that owner's per-project forecasts, so the owner number always matches the
    project list underneath it. 'Allocated' spreads the top-down GCP base across owners by those shares, so the
    owner targets add up exactly to the plan number."""
    fc, fy = dt.fc, f'FY{cf.FORECAST_YEAR} forecast'
    m = owner_map(owners, g.columns)
    billed = [p for p in g.columns if g[p].sum() >= 1]
    proj = {p: forecast_line(p, g[p], fc.horizon) for p in billed}
    info = owners.set_index('project').reindex(g.columns).fillna('')

    own_spend = g[billed].T.groupby(m[billed]).sum().T
    order = own_spend.sum().sort_values(ascending=False).index
    order = [o for o in order if o not in (SHARED_OWNER, UNASSIGNED)] + [o for o in (SHARED_OWNER, UNASSIGNED) if o in own_spend]

    lines = []
    for o in order:
        ps = [p for p in billed if m[p] == o]
        fcast = sum(proj[p].forecast for p in ps)
        lo = sum(proj[p].lo for p in ps)
        hi = sum(proj[p].hi for p in ps)
        mae = float(np.sqrt(sum(proj[p].mae ** 2 for p in ps)))      # project errors treated as independent
        exit_rr = own_spend[o].tail(3).mean()
        lf = LineForecast(o, own_spend[o], fcast, lo, hi, f'Sum of {len(ps)} project forecast{"s" if len(ps) != 1 else ""}',
                          mae, mae / exit_rr if exit_rr else np.nan, {})
        lines.append(lf)
        rows = []
        for p in sorted(ps, key=lambda p: -g[p].sum()):
            fam, env = project_family(p)
            rows.append({'Project': p, 'Family': info.loc[p, 'family'] or fam, 'Env': info.loc[p, 'environment'] or env,
                         '2026 Jan–Sep': g[p].sum(), 'Exit run-rate /mo': g[p].tail(3).mean(),
                         fy: proj[p].forecast.loc[fc.months].sum(), 'Model': proj[p].model, 'Confidence': proj[p].confidence})
        dt.owner_projects[o] = pd.DataFrame(rows).set_index('Project')
    dt.lines['GCP owners'] = lines

    t = _summarize_lines(lines, fc)
    t.insert(0, 'Projects', [len(dt.owner_projects[o]) for o in t.index])
    t.insert(2, 'Share of 2026', t['2026 Jan–Sep'] / t['2026 Jan–Sep'].sum())
    t['vs exit run-rate'] = t[fy] / (t['Exit run-rate /mo'] * 12) - 1
    base = fc.summary['fy'][mk.GCP]['Base']
    t['Allocated (ties to plan)'] = base * t[fy] / t[fy].sum()
    dt.owners = t
    dt.tables['GCP owners'] = t
    dt.totals['GCP owners'] = t[fy].sum()

    un = [p for p in billed if m[p] == UNASSIGNED]
    dt.unassigned = (pd.DataFrame({'2026 Jan–Sep': g[un].sum(), 'Exit run-rate /mo': g[un].tail(3).mean()})
                     .sort_values('2026 Jan–Sep', ascending=False) if un else pd.DataFrame())


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def chart_small_multiples(dt: Detail, key: str, color: str, cols: int = 3) -> go.Figure:
    lines = dt.lines[key]
    rows = int(np.ceil(len(lines) / cols))
    titles = [f'{fr._short(lf.name, 30)} · {lf.model}' for lf in lines]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, shared_xaxes=True,
                        vertical_spacing=0.06 if rows > 4 else 0.1, horizontal_spacing=0.06)
    band = cf._rgba(color, 0.18)
    for i, lf in enumerate(lines):
        r, c = i // cols + 1, i % cols + 1
        x_band = list(lf.forecast.index) + list(lf.forecast.index[::-1])
        fig.add_scatter(x=x_band, y=list(lf.hi) + list(lf.lo[::-1]), fill='toself', fillcolor=band, line=dict(width=0),
                        hoverinfo='skip', showlegend=False, row=r, col=c)
        fig.add_scatter(x=lf.actual.index, y=lf.actual, mode='lines+markers', line=dict(color=color, width=2),
                        marker=dict(size=5), showlegend=False, hovertemplate='%{y:$,.0f}<extra>actual</extra>', row=r, col=c)
        fig.add_scatter(x=[lf.actual.index[-1]] + list(lf.forecast.index), y=[lf.actual.iloc[-1]] + list(lf.forecast),
                        mode='lines', line=dict(color=color, width=2, dash='dash'), showlegend=False,
                        hovertemplate='%{y:$,.0f}<extra>forecast</extra>', row=r, col=c)
    fig = _layout(fig, 210 * rows + 40, legend=False)
    fig.update_layout(hovermode='closest', margin=dict(l=8, r=8, t=30, b=8))
    fig.update_annotations(font=dict(size=11, color=mk.INK2))
    fig.update_xaxes(tickformat='%b %y', nticks=4)
    fig.update_yaxes(rangemode='tozero', nticks=4)
    return fig


def chart_reconciliation(dt: Detail) -> go.Figure:
    S = dt.fc.summary
    cats = ['AWS Services', 'GCP Services']
    td = [S['fy'][mk.AWS]['Base'], S['fy'][mk.GCP]['Base']]
    bu = [dt.totals['AWS services'], dt.totals['GCP services']]
    fig = go.Figure()
    fig.add_bar(x=cats, y=td, name='Top-down (cloud total, damped trend)', marker=dict(color=mk.fr.SERIES[0], line=dict(width=0)),
                text=[_money(v) for v in td], textposition='outside', textfont=dict(color=mk.INK2))
    fig.add_bar(x=cats, y=bu, name='Bottom-up (sum of per-service models)', marker=dict(color=mk.fr.SERIES[2], line=dict(width=0)),
                text=[_money(v) for v in bu], textposition='outside', textfont=dict(color=mk.INK2))
    fig.add_bar(x=cats, y=[None, dt.totals['GCP projects']], name='Bottom-up (sum of per-project models, GCP)',
                marker=dict(color=mk.fr.SERIES[4], line=dict(width=0)),
                text=['', _money(dt.totals['GCP projects'])], textposition='outside', textfont=dict(color=mk.INK2))
    budgets = [S['fy_budget'][mk.AWS], S['fy_budget'][mk.GCP]]
    for i, b in enumerate(budgets):          # budget line spanning each cloud's group of bars
        fig.add_shape(type='line', xref='x', yref='y', x0=i - 0.45, x1=i + 0.45, y0=b, y1=b,
                      line=dict(color=mk.INK, width=2))
        fig.add_annotation(x=i + 0.45, y=b, text=f'Budget {_money(b)}', showarrow=False, xanchor='left', xshift=4,
                           font=dict(color=mk.INK, size=11))
    fig.update_layout(barmode='group')
    fig = _layout(fig, 320)
    fig.update_layout(hovermode='closest')
    fig.update_yaxes(range=[0, max(td + bu + budgets + [dt.totals['GCP projects']]) * 1.18])
    return fig


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------
def _fmt_table(t: pd.DataFrame, index_name: str, cols=None) -> str:
    t = t.copy()
    if 'FY range low' in t.columns:
        t['FY range'] = [f'{_money(a)}–{_money(b)}' for a, b in zip(t['FY range low'], t['FY range high'])]
    t = t[cols] if cols else t
    t.index = [fr._short(str(i), 44) for i in t.index]
    t.index.name = index_name
    fmts = {c: _money for c in t.columns if t[c].dtype.kind in 'fi' and c not in ('Backtest error', 'Share of 2026', 'Projects')}
    fmts['Backtest error'] = lambda v: f'±{v:.0%}' if pd.notna(v) else '—'
    fmts['Share of 2026'] = lambda v: f'{v:.0%}'
    fmts['Projects'] = lambda v: f'{int(v)}'
    fmts['Δ vs exit × 12'] = lambda v: '$0' if abs(v) < 500 else ('+' if v > 0 else '') + _money(v)
    if 'Allocated (ties to plan)' in t.columns:
        fmts['Allocated (ties to plan)'] = lambda v: f'<b>{_money(v)}</b>'
    return fr._table(t, fmts)


LEAD_COLS = ['Exit run-rate /mo', f'FY{cf.FORECAST_YEAR} forecast', 'Δ vs exit × 12', 'Model', 'Confidence']


def _reconcile_text(dt: Detail) -> str:
    S = dt.fc.summary
    parts = []
    for key, p in (('AWS services', mk.AWS), ('GCP services', mk.GCP)):
        td, bu = S['fy'][p]['Base'], dt.totals[key]
        parts.append(f'{p}: bottom-up {_money(bu)} vs top-down {_money(td)} ({bu / td - 1:+.1%})')
    return '; '.join(parts)


def _reconcile_verdict(dt: Detail) -> str:
    S = dt.fc.summary
    gaps = {p: dt.totals[k] / S['fy'][p]['Base'] - 1 for k, p in (('AWS services', mk.AWS), ('GCP services', mk.GCP))}
    worst = max(abs(v) for v in gaps.values())
    if worst <= 0.05:
        return (f'The two independent methods agree within {worst:.0%}, so the top-down base is a sound planning number.')
    hi = max(gaps, key=lambda p: abs(gaps[p]))
    return (f'{hi} differs by {gaps[hi]:+.0%}: the service-level view {"sees more" if gaps[hi] > 0 else "sees less"} growth than the '
            f'cloud total — review the ramping lines below before locking the budget.')


def _owner_bullet(dt: Detail) -> str:
    if dt.owners is None:
        return ''
    t = dt.owners
    fy = f'FY{cf.FORECAST_YEAR} forecast'
    named = t[~t.index.isin([SHARED_OWNER, UNASSIGNED])]
    covered = named['2026 Jan–Sep'].sum() / t['2026 Jan–Sep'].sum()
    al = 'Allocated (ties to plan)'
    return (f'<li><b>GCP by owner ({fy}, allocated so owners add up to the {_money(dt.fc.summary["fy"][mk.GCP]["Base"])} base):</b> '
            + ', '.join(f'{o} {_money(r[al])}' for o, r in named.iterrows()) +
            (f', shared (Support, no project) {_money(t.loc[SHARED_OWNER, al])}' if SHARED_OWNER in t.index else '') +
            (f', unassigned {_money(t.loc[UNASSIGNED, al])}' if UNASSIGNED in t.index else '') +
            f'. Named owners cover {covered:.0%} of 2026 GCP Services; the top two owners account for '
            f'{named["Share of 2026"].head(2).sum():.0%}.</li>')


def leadership_section(dt: Detail, detail_file: str | None = None) -> str:
    """Compact per-service view for the CIO/CTO report."""
    S = dt.fc.summary
    fy = f'FY{cf.FORECAST_YEAR}'
    def top(key, n=10):
        t = dt.tables[key]
        return pd.concat([t[~t.index.str.startswith('All other')].head(n), t[t.index.str.startswith('All other')]])
    link = (f'<p class="note">Full per-service and per-project detail for SVP/VP review: '
            f'<a href="{html.escape(os.path.basename(detail_file))}" style="color:var(--aws)">{html.escape(os.path.basename(detail_file))}</a></p>'
            if detail_file else '')
    gr = dt.ramping.head(3)
    return f"""
<section><h2>{fy} forecast by service — top lines</h2>
<p class="lede">Each service is forecast on its own, with the model that best predicted its own 2026 months (backtest).
Summing the services gives an independent <b>bottom-up</b> check on the cloud-level forecast above.</p>
{fr._fig_html(chart_reconciliation(dt), False)}
<ul class="ins a"><li><b>Reconciliation:</b> {_reconcile_text(dt)}. {_reconcile_verdict(dt)}</li>
<li><b>GCP projects:</b> the top {TOP_N} projects are {dt.tables['GCP projects'].iloc[:-1]['2026 Jan–Sep'].sum() / dt.tables['GCP projects']['2026 Jan–Sep'].sum():.0%} of 2026 GCP Services.
Fastest-ramping: {', '.join(f'{fr._short(n, 28)} ({_money(r["Q1 2026 avg /mo"])} → {_money(r["Exit run-rate /mo"])}/mo)' for n, r in gr.iterrows())}.</li>
{_owner_bullet(dt)}</ul>
<h3>AWS — top 10 services</h3>{_fmt_table(top('AWS services'), 'AWS service', LEAD_COLS)}
<h3>GCP — top 10 services</h3>{_fmt_table(top('GCP services'), 'GCP service', LEAD_COLS)}
<p class="note">Confidence (±% typical miss) comes from each model's backtest: how far off it was when re-predicting 1–3 months ahead inside 2026
(High &lt; 10% of run-rate, Medium &lt; 25%, Low above). Low-confidence lines are where owners' input matters most.</p>
{link}</section>"""


def _owner_section(dt: Detail) -> str:
    fy = f'FY{cf.FORECAST_YEAR} forecast'
    t = dt.owners
    S = dt.fc.summary
    chart = fr._fig_html(chart_small_multiples(dt, 'GCP owners', mk.C_PROVIDER[mk.GCP]), False)
    cols = ['Projects', '2026 Jan–Sep', 'Share of 2026', 'Exit run-rate /mo', fy, 'FY range',
            'Δ vs exit × 12', 'Allocated (ties to plan)', 'Confidence']
    summary = _fmt_table(t, 'Owner', cols)
    gap = dt.totals['GCP owners'] / S['fy'][mk.GCP]['Base'] - 1
    budget_share = ''.join(
        f'<li><b>{html.escape(o)}</b> ({int(r["Projects"])} project{"s" if r["Projects"] != 1 else ""}, '
        f'{r["Share of 2026"]:.0%} of 2026 GCP Services): exit run-rate {_money(r["Exit run-rate /mo"])}/mo → '
        f'{_money(r[fy])} bottom-up for {fy[2:6]} ({"+" if r["vs exit run-rate"] >= 0 else ""}{r["vs exit run-rate"]:.0%} vs '
        f'run-rate × 12); <b>allocated {_money(r["Allocated (ties to plan)"])}</b>; confidence {r["Confidence"]}.</li>'
        for o, r in t.iterrows())
    per_owner = ''
    for o, pt in dt.owner_projects.items():
        if pt.empty:
            continue
        pt = pt.copy()
        pt.index = [fr._short(i, 40) for i in pt.index]
        pt.index.name = 'Project'
        fm = {'2026 Jan–Sep': _money, 'Exit run-rate /mo': _money, fy: _money}
        per_owner += (f'<details><summary>{html.escape(o)} — {len(pt)} project{"s" if len(pt) != 1 else ""} · '
                      f'{_money(pt[fy].sum())} {fy[:6]}</summary>{fr._table(pt, fm)}</details>')
    un = ''
    if dt.unassigned is not None and len(dt.unassigned):
        u = dt.unassigned.copy()
        u.index.name = 'Project without an owner'
        un = (f'<h3>Still unassigned</h3>{fr._table(u, {c: _money for c in u.columns})}'
              f'<p class="note">Add an owner in <code>{html.escape(os.path.basename(OWNERS_FILE))}</code> and re-run.</p>')
    return f"""
<section><h2>GCP — by owner (SVP/VP view)</h2>
<p class="lede">Projects rolled up by owner from <code>{html.escape(os.path.basename(OWNERS_FILE))}</code>. Each owner's forecast is the
sum of their projects' forecasts (each project picks its own model), so it always matches the project list below. "{SHARED_OWNER}" is GCP Support and other charges billed to the
account rather than a project.</p>
<ul class="ins a">{budget_share}
<li>Project forecasts sum to {_money(dt.totals['GCP owners'])} vs the top-down GCP base {_money(S['fy'][mk.GCP]['Base'])} ({gap:+.0%});
the gap is growth from projects that don't exist yet. <b>Allocated</b> spreads the base across owners by their bottom-up share, so
owner targets add up exactly to the plan.</li></ul>
{chart}{summary}
<h3>Projects by owner</h3>{per_owner}{un}</section>"""


def build_detail_report(dt: Detail, out_path: str | None = None) -> str:
    fc, S = dt.fc, dt.fc.summary
    fy = f'FY{cf.FORECAST_YEAR}'
    out_path = out_path or os.path.join(mk.HERE, f'Cloud_Forecast_Detail_{fc.basis_end:%Y_%m}.html')
    cols_full = ['2026 Jan–Sep', 'Exit run-rate /mo', 'Dec 2027 /mo', f'{fy} forecast', 'FY range',
                 'Δ vs exit × 12', 'Model', 'Confidence']

    mc = dt.model_check.copy()
    mc_t = mc.pivot_table(index='Model', columns='Cloud', values='Error %').reindex(MODELS)
    mc_t.columns = [f'{c} backtest error' for c in mc_t.columns]
    mc_t.index.name = 'Model (cloud totals)'
    mc_html = fr._table(mc_t, {c: lambda v: f'±{v:.1%}' for c in mc_t.columns})

    figs = {
        'rec': chart_reconciliation(dt),            # first on the page -> carries plotly.js
        'aws': chart_small_multiples(dt, 'AWS services', mk.C_PROVIDER[mk.AWS]),
        'gcp': chart_small_multiples(dt, 'GCP services', mk.C_PROVIDER[mk.GCP]),
        'proj': chart_small_multiples(dt, 'GCP projects', mk.C_PROVIDER[mk.GCP]),
        'fam': chart_small_multiples(dt, 'GCP project families', mk.C_PROVIDER[mk.GCP]),
    }
    fh = {k: fr._fig_html(f, i == 0) for i, (k, f) in enumerate(figs.items())}

    ramp = dt.ramping.copy()
    ramp['First month billed'] = pd.to_datetime(ramp['First month billed']).dt.strftime('%b %Y')
    ramp_html = fr._table(ramp.rename_axis('Project'), {'Q1 2026 avg /mo': _money, 'Exit run-rate /mo': _money, f'{fy} forecast': _money})

    env = dt.env_split
    env_html = _fmt_table(env, 'Environment (from project name)')
    nonprod = env.loc[env.index == 'non-prod', '2026 Jan–Sep'].sum() / env['2026 Jan–Sep'].sum() if 'non-prod' in env.index else 0

    owners_html = '' if dt.owners is not None else (
        f'<p class="note">Owner roll-up: fill in <code>{html.escape(os.path.basename(OWNERS_FILE))}</code> '
        f'(owner / department per project) and re-run to get a forecast per SVP/VP.</p>')
    owner_section = _owner_section(dt) if dt.owners is not None else ''

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cloud Forecast Detail</title>
<style>{fr.css()}code{{color:var(--ink2)}}</style></head><body><main>
<header><div><h1>{fy} Cloud Forecast — Detail</h1><p class="sub">Per service and per GCP project · for SVP/VP review</p></div>
<div class="meta">Basis: {fc.basis_start:%b}–{fc.basis_end:%b %Y} actuals · Cloud Services only (Marketplace planned separately)<br>
Generated {datetime.now():%Y-%m-%d %H:%M}</div></header>

<section><h2>How to read this</h2><ul class="ins a">
<li>Every line (service, project, family) is forecast on its own. Four simple models compete — <b>Flat</b>, <b>Damped trend</b>,
<b>Linear</b> and <b>Holt (ETS)</b> exponential smoothing — and the one with the smallest error when re-predicting 2026
(trained on 5–8 months, predicting 1–3 months ahead) is used. Ties go to the simpler model.</li>
<li>Shaded bands are an approximate 80% range from that model's own backtest error, widening further out.
Forecasts are kept between {DECLINE_FLOOR:.0%} and {GROWTH_CAP:.0f}× today's run-rate (marked <i>capped</i> / <i>floored</i>);
"All other" buckets always use the damped trend. <b>Confidence</b>: High = backtest miss &lt; 10% of run-rate, Medium &lt; 25%, Low above.</li>
<li>With 9 months of history, heavier models (ARIMA, Prophet, neural nets) would overfit and cannot learn seasonality;
they become worthwhile once 2–3 years of comparable history exist per line.</li>
<li>Bottom-up totals: AWS {_money(dt.totals['AWS services'])}, GCP {_money(dt.totals['GCP services'])} vs the top-down base
AWS {_money(S['fy'][mk.AWS]['Base'])}, GCP {_money(S['fy'][mk.GCP]['Base'])}. {_reconcile_verdict(dt)}</li>
<li>GCP by project sums to {_money(dt.totals['GCP projects'])} ({dt.totals['GCP projects'] / S['fy'][mk.GCP]['Base'] - 1:+.0%} vs top-down):
most established projects have flattened, while GCP's growth comes from <b>new projects</b> appearing during the year (see
"New and ramping projects"). Read the project view as the floor if no new projects launch, and the top-down base as the
planning number.</li></ul>
{fh['rec']}
<h3>Which model fits the cloud totals best?</h3>{mc_html}
<p class="note">Error = average absolute miss predicting 1–3 months ahead within 2026, as % of the exit run-rate.</p></section>

<section><h2>AWS — top {TOP_N} services</h2>{fh['aws']}{_fmt_table(dt.tables['AWS services'], 'AWS service', cols_full)}</section>
<section><h2>GCP — top {TOP_N} services</h2>{fh['gcp']}{_fmt_table(dt.tables['GCP services'], 'GCP service', cols_full)}</section>

<section><h2>GCP — by project family</h2>
<p class="lede">Projects grouped by name with the environment removed (e.g. <code>prj-edp-prod-edw</code> + <code>prj-edp-dev-edw</code>
→ <code>prj-edp-edw</code>). Families are more stable to forecast than individual projects.</p>
{fh['fam']}{_fmt_table(dt.tables['GCP project families'], 'Project family', ['Projects'] + cols_full)}
<h3>Production vs non-production</h3>{env_html}
<p class="note">Environment is read from the project name. Non-production (dev/qa/nonprod/…) is {nonprod:.0%} of 2026 GCP Services;
"unspecified" projects have no environment in their name and could be either — adding an <code>env</code> label would close the gap.</p>
<h3>New and ramping projects</h3>{ramp_html}
<p class="note">Little or no spend in Q1 2026 and ≥ $3K/month now. These carry the most forecast risk — confirm plans with owners.</p>
{owners_html}</section>

{owner_section}
<section><h2>GCP — top {TOP_N} projects</h2>{fh['proj']}{_fmt_table(dt.tables['GCP projects'], 'GCP project', cols_full)}</section>
<p class="note">Sources: FinOps_AWS/Monthly_Cost_For_*.csv · FinOps_GCP/{mk.GCP_HISTORY}. Logic: forecast_detail.py.</p>
</main></body></html>"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    return out_path
