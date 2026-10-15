"""
2027 Cloud Budget Forecast (AWS + GCP), built ONLY from calendar-2026 actuals.

Three views, each by month and by quarter:
  a) Cloud Services only (what the AWS $500K / GCP $300K monthly budgets track)
  b) Marketplace only (third-party contracts billed through the clouds)
  c) Combined (Services + Marketplace)

Method — kept deliberately simple so it can be explained to Finance:

Cloud Services (per cloud)
  * Starting point  = average of the last 3 actual months (Jul–Sep 2026 exit run-rate).
  * Growth momentum = Theil-Sen slope of the 2026 monthly series ($ added per month). Theil-Sen is the
                      median of all pairwise slopes, so one-off months (e.g. a missing Savings Plans fee,
                      a Gemini spike) don't distort it.
  * Scenarios
      Low  — flat: spend stays at the exit run-rate.
      Base — damped trend: growth continues but loses DAMPING (15%) of its momentum each month, so
             monthly growth halves roughly every 4 months (classic damped-trend forecasting).
      High — linear: the 2026 growth slope continues undamped.
    (If the slope is negative, Low/High swap so Low is always the smaller number.)
  * AWS normalization: months where the Savings Plans fee was missing/partial (Feb–Apr 2026) get the
    typical fee added back FOR THE TREND ONLY — otherwise those artificially cheap months inflate growth.

Marketplace (per contract)
  Each third-party line is classified from its 2026 billing pattern:
      monthly   — billed in ≥5 of the 2026 months, or each of the last 3 → average of the last 3 months
      quarterly / semiannual / bimonthly — ≥2 charges with a regular gap → last amount, same cadence
      annual    — 1 charge in 2026 → same amount, same month in 2027
  Charges seen in Q4 2025 but not in 2026 are NOT in the forecast (the 2026-only basis can't see them);
  they are listed separately as renewal risk.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

import finops_exec_report as fr
import multicloud_kpis as mk
from finops_exec_report import _money, _layout

FORECAST_YEAR = 2027
BASIS_YEAR = 2026
DAMPING = 0.85
SCENARIOS = ['Low', 'Base', 'High']
MIN_CHARGE = 100          # a month counts as "billed" for a contract above this amount


@dataclass
class Forecast:
    basis_start: pd.Timestamp
    basis_end: pd.Timestamp
    horizon: pd.DatetimeIndex                       # first forecast month .. Dec FORECAST_YEAR
    budgets: dict
    svc_actual: pd.DataFrame                        # basis-year Cloud Services by provider
    mkt_actual: pd.DataFrame                        # basis-year Marketplace by provider
    svc: dict = field(default_factory=dict)         # provider -> DataFrame[Low, Base, High] over horizon
    params: dict = field(default_factory=dict)      # provider -> dict of drivers of the forecast
    drivers: dict = field(default_factory=dict)     # provider -> service-level slope table
    contracts: pd.DataFrame | None = None           # marketplace schedule (one row per contract)
    mkt: pd.DataFrame | None = None                 # horizon x provider marketplace forecast
    mkt_detail: pd.DataFrame | None = None          # horizon x (provider, contract)
    q4_risk: pd.DataFrame | None = None
    tax_rate: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    normalization: dict = field(default_factory=dict)

    @property
    def months(self) -> pd.DatetimeIndex:
        return self.horizon[self.horizon.year == FORECAST_YEAR]


# --------------------------------------------------------------------------------------
# Math
# --------------------------------------------------------------------------------------
def theil_sen(y: pd.Series) -> float:
    """Median of pairwise slopes ($ per month). Robust to one-off months."""
    v = y.to_numpy(dtype=float)
    slopes = [(v[j] - v[i]) / (j - i) for i in range(len(v)) for j in range(i + 1, len(v))]
    return float(np.median(slopes)) if slopes else 0.0


def _contract_key(provider: str, service: str) -> str:
    # Fold every Claude model / platform line into one usage-based contract.
    if 'Amazon Bedrock Edition' in service or service == 'Claude Platform':
        return 'Anthropic Claude (Bedrock usage)'
    return service


# --------------------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------------------
def forecast(df: pd.DataFrame, budgets: dict | None = None, damping: float = DAMPING) -> Forecast:
    budgets = budgets or {mk.AWS: mk.AWS_MONTHLY_BUDGET, mk.GCP: mk.GCP_MONTHLY_BUDGET}
    basis_end = min(df[df['Provider'] == p]['Date'].max() for p in (mk.AWS, mk.GCP))
    basis_start = pd.Timestamp(f'{BASIS_YEAR}-01-01')
    basis = df[(df['Date'] >= basis_start) & (df['Date'] <= basis_end)]
    horizon = pd.date_range(basis_end + relativedelta(months=1), f'{FORECAST_YEAR}-12-01', freq='MS')
    months_idx = pd.date_range(basis_start, basis_end, freq='MS')

    cat = basis.pivot_table(index='Date', columns=['Provider', 'Category'], values='Cost', aggfunc='sum').reindex(months_idx).fillna(0)
    svc_actual = cat.xs(mk.SVC, axis=1, level='Category')
    mkt_actual = cat.xs(mk.MKT, axis=1, level='Category').reindex(columns=[mk.AWS, mk.GCP], fill_value=0)
    fc = Forecast(basis_start, basis_end, horizon, budgets, svc_actual, mkt_actual)

    h = np.arange(1, len(horizon) + 1)
    damp_cum = np.cumsum(damping ** h)                      # sum_{i<=h} phi^i
    fc.normalization = {}
    for p in (mk.AWS, mk.GCP):
        s = svc_actual[p]
        trend_s = s.copy()
        if p == mk.AWS:
            sp = basis[(basis['Provider'] == p) & (basis['Service'] == fr.SP_NAME)].groupby('Date')['Cost'].sum().reindex(months_idx).fillna(0)
            typical = sp[sp > 0].median() if (sp > 0).any() else 0
            gap = (typical - sp).where(sp < 0.5 * typical, 0)
            trend_s = s + gap
            fc.normalization[p] = gap[gap > 0]
        anchor = s.tail(3).mean()
        slope = theil_sen(trend_s)
        flat = np.full(len(h), anchor)
        damped = anchor + slope * damp_cum
        linear = anchor + slope * h
        lo, hi = np.minimum.reduce([flat, damped, linear]), np.maximum.reduce([flat, damped, linear])
        fc.svc[p] = pd.DataFrame({'Low': lo, 'Base': damped, 'High': hi}, index=horizon)
        half = s.iloc[:len(s) // 2].mean()
        fc.params[p] = dict(
            ytd=s.sum(), months=len(s), anchor=anchor, slope=slope, slope_pct=slope / anchor if anchor else 0,
            first_half=half, budget=budgets[p], months_over=int((s > budgets[p]).sum()),
            peak=s.max(), peak_month=s.idxmax(), low=s.min(), low_month=s.idxmin(),
            ytd_budget=budgets[p] * len(s), base_cap=anchor + slope * damping / (1 - damping),
            raw_slope=theil_sen(s),
        )
        # service-level drivers of the slope
        sv = basis[(basis['Provider'] == p) & (basis['Category'] == mk.SVC)]
        sv = sv.pivot_table(index='Date', columns='Service', values='Cost', aggfunc='sum').reindex(months_idx).fillna(0)
        sv = sv.loc[:, (sv.mean() > 2_000) & (sv.columns != fr.SP_NAME)]
        drv = pd.DataFrame({'2026 avg / mo': sv.mean(), 'Exit run-rate': sv.tail(3).mean(),
                            'Trend $/mo per month': sv.apply(theil_sen)})
        drv['Share of growth'] = drv['Trend $/mo per month'] / slope if slope else np.nan
        fc.drivers[p] = drv.sort_values('Trend $/mo per month', ascending=False)

    _marketplace(fc, df, basis, months_idx)

    # tax rate observed in 2026 (applied to Services + Marketplace)
    for p in (mk.AWS, mk.GCP):
        tax = cat[(p, mk.TAX)].sum() if (p, mk.TAX) in cat else 0
        fc.tax_rate[p] = tax / (svc_actual[p].sum() + mkt_actual[p].sum())

    _summarize(fc)
    return fc


def contract_schedule(s: pd.Series, horizon: pd.DatetimeIndex):
    """Classify one Marketplace line's billing cadence from its basis-year months and project it over `horizon`.

    Returns (cadence, amount_per_charge, schedule Series) or None if it was never billed.
      monthly   — billed in >=5 months, or each of the last 3 -> average of the last 3 months every month
      regular   — >=2 charges with a regular gap (quarterly, semiannual, ...) -> last amount, same cadence
      annual    — 1 charge -> same amount, same month next year
    """
    billed = s[s > MIN_CHARGE]
    n = len(billed)
    if n == 0:
        return None
    sched = pd.Series(0.0, index=horizon)
    if n >= 5 or (s.tail(3) > MIN_CHARGE).all():
        cadence, amount = 'Monthly', s.tail(3).mean()
        sched[:] = amount
    elif n >= 2:
        gaps = np.diff([d.year * 12 + d.month for d in billed.index])
        k = int(round(np.median(gaps)))
        cadence = {1: 'Monthly', 2: 'Bimonthly', 3: 'Quarterly', 4: 'Every 4 months', 6: 'Semiannual'}.get(k, f'Every {k} months')
        amount = billed.iloc[-1]
        d = billed.index[-1] + relativedelta(months=k)
        while d <= horizon[-1]:
            if d in sched.index:
                sched[d] = amount
            d += relativedelta(months=k)
    else:
        cadence, amount = 'Annual', billed.iloc[-1]
        d = billed.index[-1] + relativedelta(years=1)
        if d in sched.index:
            sched[d] = amount
    return cadence, amount, sched


def _marketplace(fc: Forecast, df: pd.DataFrame, basis: pd.DataFrame, months_idx) -> None:
    m = basis[basis['Category'] == mk.MKT].copy()
    m['Contract'] = [_contract_key(p, s) for p, s in zip(m['Provider'], m['Service'])]
    piv = m.pivot_table(index='Date', columns=['Provider', 'Contract'], values='Cost', aggfunc='sum').reindex(months_idx).fillna(0)

    rows, detail = [], {}
    for (p, c), s in piv.items():
        res = contract_schedule(s, fc.horizon)
        if res is None:
            continue
        cadence, amount, sched = res
        detail[(p, c)] = sched
        fy = sched[sched.index.year == FORECAST_YEAR]
        rows.append(dict(Provider=p, Contract=c, Vendor=fr.vendor_label(c) if 'Claude' not in c else 'Anthropic',
                         Cadence=cadence, **{'Amount per charge': amount},
                         **{'2026 billed (Jan–Sep)': s.sum()},
                         **{f'Q4 {BASIS_YEAR} expected': sched[sched.index.year == BASIS_YEAR].sum()},
                         **{f'FY{FORECAST_YEAR} forecast': fy.sum()},
                         **{f'{FORECAST_YEAR} billing months': ', '.join(fy[fy > 0].index.strftime('%b')) or '—'}))
    fc.contracts = pd.DataFrame(rows).sort_values(f'FY{FORECAST_YEAR} forecast', ascending=False).reset_index(drop=True)
    fc.mkt_detail = pd.DataFrame(detail)
    fc.mkt = fc.mkt_detail.T.groupby(level=0).sum().T.reindex(columns=[mk.AWS, mk.GCP], fill_value=0)

    # Q4 renewal risk: billed in Q4 of the prior year, not billed at all in the basis year
    prev_q4 = df[(df['Category'] == mk.MKT) & (df['Date'] >= f'{BASIS_YEAR - 1}-10-01') & (df['Date'] <= f'{BASIS_YEAR - 1}-12-01')].copy()
    prev_q4['Contract'] = [_contract_key(p, s) for p, s in zip(prev_q4['Provider'], prev_q4['Service'])]
    g = prev_q4.groupby(['Provider', 'Contract', 'Date'])['Cost'].sum().reset_index()
    seen = set(zip(fc.contracts['Provider'], fc.contracts['Contract']))
    unseen = np.array([(p, c) not in seen for p, c in zip(g['Provider'], g['Contract'])], dtype=bool)
    risk = g[(g['Cost'] >= 10_000).to_numpy() & unseen]
    fc.q4_risk = risk.rename(columns={'Date': f'Billed in {BASIS_YEAR - 1}', 'Cost': 'Amount'}).sort_values('Amount', ascending=False)


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def monthly_table(fc: Forecast, view: str, scenario: str = 'Base') -> pd.DataFrame:
    """view: 'services' | 'marketplace' | 'combined' — FORECAST_YEAR months."""
    idx = fc.months
    out = pd.DataFrame(index=idx)
    if view in ('services', 'combined'):
        for p in (mk.AWS, mk.GCP):
            out[f'{p} Services'] = fc.svc[p].loc[idx, scenario]
    if view in ('marketplace', 'combined'):
        for p in (mk.AWS, mk.GCP):
            out[f'{p} Marketplace'] = fc.mkt.loc[idx, p]
    out['Total'] = out.sum(axis=1)
    if view == 'services':
        out['Budget'] = fc.budgets[mk.AWS] + fc.budgets[mk.GCP]
        out['Variance'] = out['Budget'] - out['Total']
    return out


def quarterly_table(fc: Forecast, view: str, scenario: str = 'Base') -> pd.DataFrame:
    m = monthly_table(fc, view, scenario)
    q = m.groupby(m.index.quarter).sum()
    q.index = [f'Q{i} {FORECAST_YEAR}' for i in q.index]
    q.loc[f'FY{FORECAST_YEAR}'] = q.sum()
    return q


def services_budget_table(fc: Forecast) -> pd.DataFrame:
    """Quarterly Services forecast (Low/Base/High) vs budget, per cloud and total."""
    idx = fc.months
    rows = {}
    for p in (mk.AWS, mk.GCP):
        for sc in SCENARIOS:
            rows[f'{p} — {sc}'] = fc.svc[p].loc[idx, sc]
        rows[f'{p} — Budget'] = pd.Series(fc.budgets[p], index=idx)
        rows[f'{p} — Variance (Base)'] = rows[f'{p} — Budget'] - rows[f'{p} — Base']
    for sc in SCENARIOS:
        rows[f'Total — {sc}'] = fc.svc[mk.AWS].loc[idx, sc] + fc.svc[mk.GCP].loc[idx, sc]
    rows['Total — Budget'] = pd.Series(fc.budgets[mk.AWS] + fc.budgets[mk.GCP], index=idx)
    rows['Total — Variance (Base)'] = rows['Total — Budget'] - rows['Total — Base']
    t = pd.DataFrame(rows)
    q = t.groupby(t.index.quarter).sum()
    q.index = [f'Q{i}' for i in q.index]
    q.loc[f'FY{FORECAST_YEAR}'] = q.sum()
    return q.T


def _ftable(df: pd.DataFrame, index_name: str = '', var_rows=('Variance',), var_cols=('Variance',)) -> str:
    """Money table; cells in variance rows/cols are green when ≥ 0 (under budget), red when < 0."""
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in [index_name] + list(df.columns))
    body = []
    for idx, row in df.iterrows():
        label = idx.strftime('%b %Y') if isinstance(idx, pd.Timestamp) else str(idx)
        is_var_row = any(v in label for v in var_rows)
        strong = label.startswith(('FY', 'Total')) or 'Base' in label
        cls_row = ' class="total"' if label.startswith('FY') else ''
        cells = [f'<td>{"<b>" + html.escape(label) + "</b>" if strong else html.escape(label)}</td>']
        for c, v in row.items():
            is_var = is_var_row or any(str(c).startswith(x) for x in var_cols)
            cls = (' class="good"' if v >= 0 else ' class="bad"') if is_var and abs(v) >= 0.5 else ''
            txt = ('+' if is_var and v > 0 else '') + _money(v)
            cells.append(f'<td{cls}>{txt}</td>')
        body.append(f'<tr{cls_row}>' + ''.join(cells) + '</tr>')
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# --------------------------------------------------------------------------------------
# Executive summary
# --------------------------------------------------------------------------------------
def _summarize(fc: Forecast) -> None:
    S = fc.summary
    idx = fc.months
    fy = {p: {sc: fc.svc[p].loc[idx, sc].sum() for sc in SCENARIOS} for p in (mk.AWS, mk.GCP)}
    fy_budget = {p: fc.budgets[p] * 12 for p in (mk.AWS, mk.GCP)}
    mkt_fy = {p: fc.mkt.loc[idx, p].sum() for p in (mk.AWS, mk.GCP)}
    S.update(fy=fy, fy_budget=fy_budget, mkt_fy=mkt_fy)
    S['svc_total'] = {sc: fy[mk.AWS][sc] + fy[mk.GCP][sc] for sc in SCENARIOS}
    S['budget_total'] = sum(fy_budget.values())
    S['mkt_total'] = sum(mkt_fy.values())
    S['combined'] = {sc: S['svc_total'][sc] + S['mkt_total'] for sc in SCENARIOS}
    S['tax'] = sum((fy[p]['Base'] + mkt_fy[p]) * fc.tax_rate[p] for p in (mk.AWS, mk.GCP))
    q4 = fc.horizon[fc.horizon.year == BASIS_YEAR]
    S['q4_basis'] = {p: fc.svc[p].loc[q4, 'Base'].sum() + fc.mkt.loc[q4, p].sum() for p in (mk.AWS, mk.GCP)}
    S['first_over'] = {}
    for p in (mk.AWS, mk.GCP):
        over = fc.svc[p].loc[idx, 'Base'] > fc.budgets[p]
        S['first_over'][p] = over.idxmax() if over.any() else None

    mn = f'{fc.basis_start:%b}–{fc.basis_end:%b %Y}'
    paras = []

    # 1. Starting point
    lines = []
    for p in (mk.AWS, mk.GCP):
        pr = fc.params[p]
        gap = pr['budget'] - pr['anchor']
        lines.append(f'<b>{p}</b> Cloud Services totaled {_money(pr["ytd"])} for {mn} against a {_money(pr["ytd_budget"])} '
                     f'pro-rated budget ({_money(pr["ytd_budget"] - pr["ytd"])} {"under" if pr["ytd"] <= pr["ytd_budget"] else "over"}). '
                     f'The exit run-rate (last 3 months) is <b>{_money(pr["anchor"])}/month</b> — '
                     f'{pr["anchor"] / pr["budget"]:.0%} of the {_money(pr["budget"])} monthly budget, '
                     + (f'leaving {_money(gap)}/month of headroom.' if gap >= 0 else
                        f'<span class="bad">already {_money(-gap)}/month over budget</span>; '
                        f'{pr["months_over"]} of {pr["months"]} months in {BASIS_YEAR} exceeded budget.'))
    paras.append(('Where we start', lines))

    # 2. Growth pattern
    lines = []
    for p in (mk.AWS, mk.GCP):
        pr = fc.params[p]
        lines.append(f'<b>{p}</b> added about <b>{_money(pr["slope"])} per month</b> in {BASIS_YEAR} '
                     f'({pr["slope_pct"]:.1%} of run-rate per month; first-half average {_money(pr["first_half"])} → exit '
                     f'{_money(pr["anchor"])}, {pr["anchor"] / pr["first_half"] - 1:+.0%}). '
                     f'Range {_money(pr["low"])} ({pr["low_month"]:%b}) to {_money(pr["peak"])} ({pr["peak_month"]:%b}).')
        gap = fc.normalization.get(p)
        if gap is not None and len(gap):
            lines.append(f'Normalization: no/partial Savings Plans fee was billed in {", ".join(gap.index.strftime("%b"))} '
                         f'(≈ {_money(gap.sum())} in total), which made those months look cheaper. The typical fee is added back '
                         f'for the trend only — this lowers the {p} growth estimate from {_money(pr["raw_slope"])} to '
                         f'{_money(pr["slope"])} per month.')
        d = fc.drivers[p]
        up = d[d['Trend $/mo per month'] > 0].head(4)
        if len(up):
            lines.append(f'{p} growth drivers: ' + ', '.join(
                f'{fr._short(s, 30)} (+{_money(r["Trend $/mo per month"])}/mo each month, now {_money(r["Exit run-rate"])}/mo)'
                for s, r in up.iterrows()) + '.')
        down = d[d['Trend $/mo per month'] < -500].tail(2)
        if len(down):
            lines.append(f'{p} offsets: ' + ', '.join(
                f'{fr._short(s, 30)} ({_money(r["Trend $/mo per month"])}/mo each month)' for s, r in down.iterrows()) + '.')
    paras.append((f'{BASIS_YEAR} spend pattern', lines))

    # 3. The forecast
    lines = []
    for p in (mk.AWS, mk.GCP):
        pr, f = fc.params[p], fy[p]
        dec = fc.svc[p].loc[idx[-1]]
        var = fy_budget[p] - f['Base']
        fo = S['first_over'][p]
        lines.append(
            f'<b>{p} Services FY{FORECAST_YEAR}: {_money(f["Base"])} base</b> (low {_money(f["Low"])} – high {_money(f["High"])}) '
            f'vs {_money(fy_budget[p])} budget → <span class="{"good" if var >= 0 else "bad"}">'
            f'{_money(abs(var))} {"under" if var >= 0 else "over"}</span>. '
            f'Base run-rate reaches {_money(dec["Base"])}/month by Dec {FORECAST_YEAR} (ceiling ≈ {_money(pr["base_cap"])}). '
            + (f'Base crosses the {_money(pr["budget"])} monthly budget in {fo:%b %Y}.' if fo is not None and pr['anchor'] <= pr['budget']
               else (f'Even the flat (low) scenario is {_money(f["Low"] - fy_budget[p])} over budget.' if f['Low'] > fy_budget[p] else
                     f'Base stays within the monthly budget all year.' if fo is None else '')))
    lines.append(f'Combined Services: <b>{_money(S["svc_total"]["Base"])}</b> base vs {_money(S["budget_total"])} budget '
                 f'(range {_money(S["svc_total"]["Low"])} – {_money(S["svc_total"]["High"])}).')
    paras.append((f'FY{FORECAST_YEAR} Cloud Services outlook', lines))

    # 4. Marketplace
    c = fc.contracts
    fycol = f'FY{FORECAST_YEAR} forecast'
    lines = [f'Marketplace FY{FORECAST_YEAR}: <b>{_money(S["mkt_total"])}</b> '
             f'(AWS {_money(mkt_fy[mk.AWS])}, GCP {_money(mkt_fy[mk.GCP])}) across {int((c[fycol] > 0).sum())} contracts, '
             f'vs {_money(fc.mkt_actual.sum().sum())} billed {mn}.']
    for cad in ['Quarterly', 'Annual', 'Monthly', 'Semiannual', 'Bimonthly']:
        sub = c[(c['Cadence'] == cad) & (c[fycol] > 0)]
        if len(sub):
            months_col = f'{FORECAST_YEAR} billing months'
            dup = c['Vendor'].duplicated(keep=False)
            label = np.where(dup.loc[sub.index], [fr._short(x, 30) for x in sub['Contract']], sub['Vendor'])
            lines.append(f'{cad} ({_money(sub[fycol].sum())}): ' + ', '.join(
                f'{v} {_money(amt)}' + ('/mo' if cad == 'Monthly' else f' ({mo})' if cad == 'Annual' else f' × {len(mo.split(", "))} ({mo})')
                for v, amt, mo in zip(label, sub['Amount per charge'], sub[months_col])) + '.')
    q = quarterly_table(fc, 'marketplace')['Total']
    lines.append('By quarter: ' + ', '.join(f'{i} {_money(v)}' for i, v in q.items() if not i.startswith('FY')) +
                 f' — heaviest quarter is {q.drop(f"FY{FORECAST_YEAR}").idxmax()}.')
    paras.append(('Marketplace commitments', lines))

    # 5. Combined + tax
    paras.append(('Total cloud funding need', [
        f'Services (base) + Marketplace = <b>{_money(S["combined"]["Base"])}</b> for FY{FORECAST_YEAR} '
        f'(range {_money(S["combined"]["Low"])} – {_money(S["combined"]["High"])}).',
        f'Estimated tax at the {BASIS_YEAR} effective rate (AWS {fc.tax_rate[mk.AWS]:.1%}, GCP {fc.tax_rate[mk.GCP]:.1%}): '
        f'≈ {_money(S["tax"])}, bringing the all-in estimate to ≈ <b>{_money(S["combined"]["Base"] + S["tax"])}</b>.',
        f'Remaining {BASIS_YEAR} (Q4) outlook on the same method: AWS {_money(S["q4_basis"][mk.AWS])}, '
        f'GCP {_money(S["q4_basis"][mk.GCP])} (Services + Marketplace).',
    ]))

    # 6. Risks / what's not in the numbers
    risk = []
    if len(fc.q4_risk):
        risk.append(f'Q4 renewals billed in Q4 {BASIS_YEAR - 1} but not seen in {BASIS_YEAR} are <b>not</b> in the forecast '
                    f'({_money(fc.q4_risk["Amount"].sum())}): ' + ', '.join(
                        f'{r.Provider} {fr._short(r.Contract, 30)} {_money(r.Amount)}' for r in fc.q4_risk.itertuples()) + '.')
    risk += [
        f'Only {fc.params[mk.AWS]["months"]} months of {BASIS_YEAR} data — month-of-year seasonality cannot be estimated, so every '
        f'{FORECAST_YEAR} month follows the trend line.',
        'Annual Marketplace lines are assumed to renew at the same price; any one-off purchase in 2026 (e.g. services engagements) '
        'is included unless removed from the contract table.',
        'Prepaid commitments and one-time charges (like the Oct 2025 VMware Engine charge) are not predictable from the trend.',
        'Usage-based GenAI (Gemini API, Bedrock/Claude) is growing fast and can move the GCP/AWS trend quickly.',
    ]
    paras.append(('Assumptions & risks', risk))

    # 7. Recommendation
    rec = []
    for p in (mk.AWS, mk.GCP):
        gap = fy[p]['Base'] - fy_budget[p]
        if gap > 0:
            rec.append(f'{p}: to hold the {_money(fc.budgets[p])}/month budget, Services need ≈ {_money(gap / 12)}/month of savings '
                       f'vs the base forecast ({_money(gap)} for the year) — or a budget of ≈ {_money(fy[p]["Base"] / 12)}/month.')
        else:
            rec.append(f'{p}: base forecast fits the {_money(fc.budgets[p])}/month budget with {_money(-gap)} to spare; '
                       f'the high scenario would need {_money(fy[p]["High"] / 12)}/month.')
    rec.append(f'Plan Marketplace separately at ≈ {_money(S["mkt_total"])} with the renewal calendar below, and decide on the '
               f'{_money(fc.q4_risk["Amount"].sum()) if len(fc.q4_risk) else "$0"} of Q4-type renewals not visible in {BASIS_YEAR}.')
    paras.append(('Recommendation', rec))
    S['paragraphs'] = paras


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})'


def chart_services(fc: Forecast, p: str) -> go.Figure:
    col = mk.C_PROVIDER[p]
    act = fc.svc_actual[p]
    f = fc.svc[p]
    x_band = list(f.index) + list(f.index[::-1])
    fig = go.Figure()
    fig.add_scatter(x=x_band, y=list(f['High']) + list(f['Low'][::-1]), fill='toself', fillcolor=_rgba(col, 0.18),
                    line=dict(width=0), name='Low–High range', hoverinfo='skip')
    fig.add_scatter(x=act.index, y=act, name=f'{BASIS_YEAR} actual', mode='lines+markers',
                    line=dict(color=col, width=2), marker=dict(size=7), hovertemplate='%{y:$,.0f}')
    fig.add_scatter(x=[act.index[-1]] + list(f.index), y=[act.iloc[-1]] + list(f['Base']), name='Base forecast',
                    mode='lines+markers', line=dict(color=col, width=2, dash='dash'), marker=dict(size=5),
                    hovertemplate='%{y:$,.0f}')
    for sc in ('Low', 'High'):
        fig.add_scatter(x=f.index, y=f[sc], name=sc, mode='lines', line=dict(width=0), showlegend=False,
                        hovertemplate=f'{sc} ' + '%{y:$,.0f}<extra></extra>')
    fig.add_hline(y=fc.budgets[p], line=dict(color=mk.INK, width=1.5),
                  annotation=dict(text=f'{p} budget {_money(fc.budgets[p])}/mo', font=dict(color=mk.INK, size=11)),
                  annotation_position='top left')
    fig.add_vline(x=fc.horizon[0] - pd.Timedelta(days=15), line=dict(color=mk.AXIS, width=1))
    fig.add_annotation(x=fc.horizon[0], y=1, yref='paper', text='Forecast →', showarrow=False, xanchor='left',
                       font=dict(color=mk.MUTED, size=11))
    last = f.iloc[-1]
    fig.add_annotation(x=f.index[-1], y=last['Base'], text=f'{_money(last["Base"])}', showarrow=False, xanchor='left',
                       xshift=8, font=dict(color=mk.INK, size=11))
    fig = _layout(fig, 360)
    fig.update_layout(margin=dict(r=70))
    ymax = max(f['High'].max(), fc.budgets[p], act.max()) * 1.08
    fig.update_yaxes(range=[0, ymax])
    return fig


def chart_marketplace(fc: Forecast) -> go.Figure:
    act, f = fc.mkt_actual, fc.mkt
    det = fc.mkt_detail
    fig = go.Figure()
    for p in (mk.GCP, mk.AWS):
        col = mk.COLORS[(p, mk.MKT)]
        fig.add_bar(x=act.index, y=act[p], name=f'{p} Marketplace (actual)', marker=dict(color=col, line=dict(width=0)),
                    hovertemplate='%{y:$,.0f}')
        names = [', '.join(f'{fr._short(c, 22)}' for (pp, c), v in det.loc[d].items() if pp == p and v > 1000) for d in f.index]
        fig.add_bar(x=f.index, y=f[p], name=f'{p} Marketplace (forecast)', customdata=names,
                    marker=dict(color=_rgba(col, 0.45), line=dict(color=col, width=1)),
                    hovertemplate='%{y:$,.0f}<br>%{customdata}<extra></extra>')
    fig.update_layout(barmode='stack')
    fig.add_vline(x=fc.horizon[0] - pd.Timedelta(days=15), line=dict(color=mk.AXIS, width=1))
    fig.add_annotation(x=fc.horizon[0], y=1, yref='paper', text='Forecast →', showarrow=False, xanchor='left',
                       font=dict(color=mk.MUTED, size=11))
    return _layout(fig, 380)


def chart_combined(fc: Forecast) -> go.Figure:
    fig = go.Figure()
    series = [
        ((mk.GCP, mk.SVC), fc.svc_actual[mk.GCP], fc.svc[mk.GCP]['Base']),
        ((mk.AWS, mk.SVC), fc.svc_actual[mk.AWS], fc.svc[mk.AWS]['Base']),
        ((mk.GCP, mk.MKT), fc.mkt_actual[mk.GCP], fc.mkt[mk.GCP]),
        ((mk.AWS, mk.MKT), fc.mkt_actual[mk.AWS], fc.mkt[mk.AWS]),
    ]
    for (p, c), a, f in series:
        col = mk.COLORS[(p, c)]
        fig.add_bar(x=list(a.index) + list(f.index), y=list(a) + list(f), name=f'{p} {c}',
                    marker=dict(color=[col] * len(a) + [_rgba(col, 0.5)] * len(f), line=dict(width=0)),
                    hovertemplate='%{y:$,.0f}')
    fig.update_layout(barmode='stack')
    total_budget = fc.budgets[mk.AWS] + fc.budgets[mk.GCP]
    fig.add_hline(y=total_budget, line=dict(color=mk.INK, width=1.5),
                  annotation=dict(text=f'Services budget (AWS+GCP) {_money(total_budget)}/mo', font=dict(color=mk.INK, size=11)),
                  annotation_position='top left')
    fig.add_vline(x=fc.horizon[0] - pd.Timedelta(days=15), line=dict(color=mk.AXIS, width=1))
    fig.add_annotation(x=fc.horizon[0], y=1, yref='paper', text='Forecast (base) →', showarrow=False, xanchor='left',
                       font=dict(color=mk.MUTED, size=11))
    return _layout(fig, 420)


def chart_fy_bridge(fc: Forecast) -> go.Figure:
    """FY forecast vs budget per cloud (Services) + Marketplace, base with low/high error bars."""
    S = fc.summary
    cats = [f'AWS Services', f'GCP Services', 'Marketplace']
    base = [S['fy'][mk.AWS]['Base'], S['fy'][mk.GCP]['Base'], S['mkt_total']]
    lo = [S['fy'][mk.AWS]['Low'], S['fy'][mk.GCP]['Low'], S['mkt_total']]
    hi = [S['fy'][mk.AWS]['High'], S['fy'][mk.GCP]['High'], S['mkt_total']]
    budget = [S['fy_budget'][mk.AWS], S['fy_budget'][mk.GCP], None]
    cols = [mk.C_PROVIDER[mk.AWS], mk.C_PROVIDER[mk.GCP], mk.COLORS[(mk.GCP, mk.MKT)]]
    fig = go.Figure()
    fig.add_bar(x=cats, y=base, name=f'FY{FORECAST_YEAR} base forecast', marker=dict(color=cols, line=dict(width=0)),
                error_y=dict(type='data', symmetric=False, array=[h - b for h, b in zip(hi, base)],
                             arrayminus=[b - l for b, l in zip(base, lo)], color=mk.INK2, thickness=1.5, width=8),
                text=[_money(v) for v in base], textposition='inside', insidetextanchor='end',
                textfont=dict(color='#ffffff', size=12),
                hovertemplate='%{x}: %{y:$,.0f}<extra></extra>')
    fig.add_scatter(x=cats[:2], y=budget[:2], name='FY budget', mode='markers',
                    marker=dict(symbol='line-ew', size=60, line=dict(width=3, color=mk.INK)),
                    hovertemplate='Budget %{y:$,.0f}<extra></extra>')
    fig = _layout(fig, 340)
    fig.update_layout(hovermode='closest')
    return fig


# --------------------------------------------------------------------------------------
# HTML section (embedded in the Cloud KPI report)
# --------------------------------------------------------------------------------------
def html_section(fc: Forecast) -> str:
    S = fc.summary
    fyl = f'FY{FORECAST_YEAR}'

    def tile(label, value, detail):
        return f'<div class="tile"><div class="k">{label}</div><div class="v">{value}</div><div class="d">{detail}</div></div>'

    def vs(v, b):
        d = b - v
        return f'<span class="{"good" if d >= 0 else "bad"}">{_money(abs(d))} {"under" if d >= 0 else "over"}</span> {_money(b)} budget'

    tiles = '<div class="tiles">' + ''.join([
        tile(f'{fyl} Cloud Services · base', _money(S['svc_total']['Base']), vs(S['svc_total']['Base'], S['budget_total'])),
        tile(f'<span class="dot" style="background:{mk.C_PROVIDER[mk.AWS]}"></span>AWS Services', _money(S['fy'][mk.AWS]['Base']),
             vs(S['fy'][mk.AWS]['Base'], S['fy_budget'][mk.AWS])),
        tile(f'<span class="dot" style="background:{mk.C_PROVIDER[mk.GCP]}"></span>GCP Services', _money(S['fy'][mk.GCP]['Base']),
             vs(S['fy'][mk.GCP]['Base'], S['fy_budget'][mk.GCP])),
        tile(f'{fyl} Marketplace', _money(S['mkt_total']), f'AWS {_money(S["mkt_fy"][mk.AWS])} · GCP {_money(S["mkt_fy"][mk.GCP])}'),
        tile(f'{fyl} Combined · base', _money(S['combined']['Base']),
             f'range {_money(S["combined"]["Low"])} – {_money(S["combined"]["High"])}'),
        tile('All-in incl. est. tax', _money(S['combined']['Base'] + S['tax']), f'tax ≈ {_money(S["tax"])}'),
    ]) + '</div>'

    summary = ''.join(f'<h3>{html.escape(t)}</h3><ul class="ins a">' + ''.join(f'<li>{l}</li>' for l in lines) + '</ul>'
                      for t, lines in S['paragraphs'])

    method = (f'<details><summary>How the forecast is built</summary><ul class="ins a">'
              f'<li><b>Basis:</b> only {fc.basis_start:%b}–{fc.basis_end:%b %Y} actuals. Forecast covers {fc.horizon[0]:%b %Y}–Dec {FORECAST_YEAR}.</li>'
              f'<li><b>Starting point:</b> average of the last 3 actual months (exit run-rate).</li>'
              f'<li><b>Trend:</b> Theil-Sen slope of the {BASIS_YEAR} monthly series — the median of all month-to-month '
              f'pairwise slopes, so one unusual month does not swing it.</li>'
              f'<li><b>Low</b> = flat at the exit run-rate · <b>Base</b> = damped trend (momentum fades {1 - DAMPING:.0%} per month, '
              f'halving roughly every 4 months) · <b>High</b> = the {BASIS_YEAR} slope continues unchanged.</li>'
              f'<li><b>Marketplace:</b> each contract repeats its {BASIS_YEAR} billing cadence and last amount '
              f'(monthly = last-3-month average).</li>'
              f'<li><b>Tax:</b> {BASIS_YEAR} effective tax rate applied to Services + Marketplace, shown separately.</li></ul></details>')

    c = fc.contracts.copy()
    mcol = f'{FORECAST_YEAR} billing months'
    c[mcol] = np.where(c['Cadence'].eq('Monthly'), 'Every month', c[mcol])
    dup = c['Vendor'].duplicated(keep=False)
    c['Contract'] = [fr._short(x, 34) if d else v for x, v, d in zip(c['Contract'], c['Vendor'], dup)]
    c = c.set_index('Contract').drop(columns=['Vendor'])
    c.index.name = 'Marketplace contract'
    ct = fr._table(c, {'Amount per charge': _money, '2026 billed (Jan–Sep)': _money, f'Q4 {BASIS_YEAR} expected': _money,
                        f'{fyl} forecast': _money})
    risk = ''
    if len(fc.q4_risk):
        r = fc.q4_risk.copy()
        r[f'Billed in {BASIS_YEAR - 1}'] = pd.to_datetime(r[f'Billed in {BASIS_YEAR - 1}']).dt.strftime('%b %Y')
        r = r.set_index('Contract')
        r.index.name = f'Not in forecast — billed Q4 {BASIS_YEAR - 1}, not in {BASIS_YEAR}'
        risk = fr._table(r, {'Amount': _money})

    drivers = ''
    for p in (mk.AWS, mk.GCP):
        d = fc.drivers[p].head(8).copy()
        d.index = [fr._short(s, 40) for s in d.index]
        d.index.name = f'{p} growth drivers ({BASIS_YEAR})'
        drivers += '<div>' + fr._table(d, {'2026 avg / mo': _money, 'Exit run-rate': _money,
                                            'Trend $/mo per month': lambda v: ('+' if v > 0 else '') + _money(v),
                                            'Share of growth': lambda v: f'{v:.0%}' if pd.notna(v) else '—'}) + '</div>'

    figs = [chart_fy_bridge(fc), chart_services(fc, mk.AWS), chart_services(fc, mk.GCP), chart_marketplace(fc), chart_combined(fc)]
    fh = [fr._fig_html(f, False) for f in figs]

    def mt(view):
        t = monthly_table(fc, view)
        t.index = t.index.strftime('%b %Y')
        return _ftable(t, 'Month')

    return f"""
<section><h2>{fyl} forecast — executive summary</h2>
<p class="lede">Built only from {fc.basis_start:%b}–{fc.basis_end:%b %Y} actuals. Services are compared to the monthly budgets
(AWS {_money(fc.budgets[mk.AWS])}, GCP {_money(fc.budgets[mk.GCP])}); Marketplace is planned separately.</p>
{tiles}{fh[0]}{summary}{method}</section>

<section><h2>a) Cloud Services only — {fyl} forecast</h2>
<p class="lede">Solid = {BASIS_YEAR} actuals · dashed = base forecast · shaded = low–high range.</p>
<h3>AWS</h3>{fh[1]}<h3>GCP</h3>{fh[2]}
<h3>By quarter — scenarios vs budget</h3>{_ftable(services_budget_table(fc), 'Cloud Services', var_cols=())}
<details><summary>By month (base)</summary>{mt('services')}</details>
<div class="cols">{drivers}</div></section>

<section><h2>b) Marketplace only — {fyl} forecast</h2>
<p class="lede">Solid = {BASIS_YEAR} actuals · light = forecast from each contract's billing cadence. Hover for contract names.</p>{fh[3]}
<div class="cols"><div><h3>By quarter</h3>{_ftable(quarterly_table(fc, 'marketplace'), 'Quarter')}</div>
<div><h3>By month</h3>{mt('marketplace')}</div></div>
<h3>Contract schedule</h3>{ct}{('<h3>Renewal risk</h3>' + risk) if risk else ''}</section>

<section><h2>c) Combined — Services + Marketplace, {fyl} forecast (base)</h2>{fh[4]}
<h3>By quarter</h3>{_ftable(quarterly_table(fc, 'combined'), 'Quarter')}
<h3>By month</h3>{mt('combined')}
<p class="note">Excludes tax (≈ {_money(S['tax'])} estimated for {fyl}).</p></section>
"""
