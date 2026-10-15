"""
AWS FinOps by Linked Account — 2026 actuals and a 2027 forecast built ONLY from 2026.

Queries Cost Explorer once from the management (payer / roll-up) account
    Entercom Communications · 724972922289
grouped by LINKED_ACCOUNT × SERVICE, so every member account in the organization is included, plus a
SERVICE × BILLING_ENTITY query that splits AWS Services from AWS Marketplace authoritatively.

Forecast (same method as cloud_forecast.py, so numbers are comparable):
  Cloud Services  start = Jul–Sep exit run-rate · trend = Theil-Sen slope of 2026 (Savings Plans gaps normalized)
                  Low = flat · Base = damped trend (momentum fades 15%/month) · High = trend continues
  Per account     each linked account is forecast on its own with the backtested model competition from
                  forecast_detail.py; the accounts are then summed as an independent bottom-up check
  Marketplace     each (account, product) repeats its 2026 billing cadence and amount (monthly / quarterly / annual)

Usage
-----
    python AWS_FinOps_By_Account.py              # SSO login in the browser, pull 2026, build the report
    python AWS_FinOps_By_Account.py --cached     # rebuild from AWS_2026_By_Account.csv without calling AWS
"""
from __future__ import annotations

import argparse
import html
import os
import webbrowser
from dataclasses import dataclass, field
from datetime import date, datetime
from time import sleep
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

import cloud_forecast as cf
import finops_exec_report as fr
import forecast_detail as fd
from finops_exec_report import _money, _layout

HERE = os.path.dirname(os.path.abspath(__file__))

# Roll-up (management / payer) account — the one Cost Explorer sees the whole organization from
ROLLUP_ACCOUNT_NAME = 'Entercom Communications'
ROLLUP_ACCOUNT_ID = '724972922289'
SSO_START_URL = 'https://d-906760626e.awsapps.com/start'
SSO_REGION = 'us-east-1'

BASIS_YEAR = 2026
FORECAST_YEAR = 2027
MONTHLY_BUDGET = 500_000                  # AWS Cloud Services (Marketplace and Tax excluded)
METRIC = 'NetUnblendedCost'               # same metric as FinOps.ipynb
CACHE_FILE = os.path.join(HERE, f'AWS_{BASIS_YEAR}_By_Account.csv')
TOP_ACCOUNTS = 15

SVC, MKT, TAX = 'Cloud Services', 'Marketplace', 'Tax'
C_SVC, C_MKT = '#d95926', '#c98500'       # AWS orange / AWS Marketplace yellow (dark-theme steps)


# --------------------------------------------------------------------------------------
# Authentication (same SSO device flow as FinOps.ipynb)
# --------------------------------------------------------------------------------------
def sso_login(start_url: str = SSO_START_URL, region: str = SSO_REGION):
    """Open the browser for AWS SSO approval and wait for it. Returns (boto3 Session, token dict)."""
    from boto3.session import Session
    session = Session()
    oidc = session.client('sso-oidc', region_name=region)
    client = oidc.register_client(clientName='finops-by-account', clientType='public')
    auth = oidc.start_device_authorization(clientId=client['clientId'], clientSecret=client['clientSecret'], startUrl=start_url)
    print('Approve the AWS sign-in in your browser:', auth['verificationUriComplete'])
    webbrowser.open(auth['verificationUriComplete'], autoraise=True)
    for _ in range(auth['expiresIn'] // auth['interval'] + 1):
        sleep(auth['interval'])
        try:
            token = oidc.create_token(grantType='urn:ietf:params:oauth:grant-type:device_code', deviceCode=auth['deviceCode'],
                                      clientId=client['clientId'], clientSecret=client['clientSecret'])
            return session, token
        except oidc.exceptions.AuthorizationPendingException:
            continue
        except oidc.exceptions.SlowDownException:
            sleep(auth['interval'])
    raise TimeoutError('AWS SSO login was not approved in time')


def rollup_session(session, token, account_id: str = ROLLUP_ACCOUNT_ID, region: str = SSO_REGION):
    """boto3 Session with role credentials for the roll-up account (prefers a billing role)."""
    from boto3.session import Session
    sso = session.client('sso', region_name=region)
    roles = [r['roleName'] for r in sso.list_account_roles(accessToken=token['accessToken'], accountId=account_id)['roleList']]
    # least privilege first: Cost Explorer only needs read access
    rank = lambda r: (0 if 'ReadOnly' in r and 'Billing' in r else 1 if 'ReadOnly' in r or r == 'MarketplaceRead'
                      else 2 if 'Billing' in r else 3)
    role = sorted(roles, key=rank)[0]
    if role == 'MarketplaceRead':                     # same substitution FinOps.ipynb makes
        role = 'BillingReadOnly'
    creds = sso.get_role_credentials(roleName=role, accountId=account_id, accessToken=token['accessToken'])['roleCredentials']
    print(f'Using role {role} on {account_id}')
    return Session(region_name='us-east-1', aws_access_key_id=creds['accessKeyId'],
                   aws_secret_access_key=creds['secretAccessKey'], aws_session_token=creds['sessionToken'])


# --------------------------------------------------------------------------------------
# Cost Explorer
# --------------------------------------------------------------------------------------
def _query(ce, start: str, end: str, group_by: list):
    """All pages of a MONTHLY get_cost_and_usage; returns (rows, dimension attributes)."""
    rows, attrs, token = [], {}, None
    while True:
        kw = dict(TimePeriod={'Start': start, 'End': end}, Granularity='MONTHLY', Metrics=[METRIC],
                  GroupBy=[{'Type': 'DIMENSION', 'Key': k} for k in group_by])
        if token:
            kw['NextPageToken'] = token
        resp = ce.get_cost_and_usage(**kw)
        for a in resp.get('DimensionValueAttributes', []):
            attrs[a['Value']] = a.get('Attributes', {}).get('description', '')
        for r in resp['ResultsByTime']:
            for g in r['Groups']:
                rows.append((r['TimePeriod']['Start'], *g['Keys'], float(g['Metrics'][METRIC]['Amount'])))
        token = resp.get('NextPageToken')
        if not token:
            return rows, attrs


def fetch_year(aws_session, year: int = BASIS_YEAR, include_current_month: bool = False) -> pd.DataFrame:
    """Monthly cost by linked account × service for `year` (complete months only unless include_current_month)."""
    ce = aws_session.client('ce', region_name='us-east-1')
    start = date(year, 1, 1)
    first_this_month = date.today().replace(day=1)
    end = min(date(year + 1, 1, 1), date.today() + relativedelta(days=1) if include_current_month else first_this_month)
    s, e = start.isoformat(), end.isoformat()                 # End is exclusive in Cost Explorer

    acct_rows, names = _query(ce, s, e, ['LINKED_ACCOUNT', 'SERVICE'])
    ent_rows, _ = _query(ce, s, e, ['SERVICE', 'BILLING_ENTITY'])

    df = pd.DataFrame(acct_rows, columns=['Date', 'AccountId', 'Service', 'Cost'])
    ent = pd.DataFrame(ent_rows, columns=['Date', 'Service', 'BillingEntity', 'Cost'])
    # a service's billing entity = the entity carrying most of its cost
    entity = ent.groupby(['Service', 'BillingEntity'])['Cost'].sum().abs().reset_index().sort_values('Cost')
    entity = entity.drop_duplicates('Service', keep='last').set_index('Service')['BillingEntity']

    df['Date'] = pd.to_datetime(df['Date'])
    df['AccountName'] = df['AccountId'].map(names).replace('', np.nan).fillna(df['AccountId'])
    df['BillingEntity'] = df['Service'].map(entity)
    df['Category'] = [TAX if sv == 'Tax' else MKT if isinstance(be, str) and 'Marketplace' in be
                      else (SVC if isinstance(be, str) else {fr.AWS: SVC, fr.MKT: MKT, fr.TAX: TAX}[fr.classify_service(sv)])
                      for sv, be in zip(df['Service'], df['BillingEntity'])]
    df['Partial'] = df['Date'] >= pd.Timestamp(first_this_month)
    df['PulledAt'] = datetime.now().isoformat(timespec='seconds')
    return df[['Date', 'AccountId', 'AccountName', 'Service', 'BillingEntity', 'Category', 'Cost', 'Partial', 'PulledAt']]


def load_cached(path: str = CACHE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'AccountId': str}, parse_dates=['Date'])
    df['Service'] = df['Service'].replace(fr.SERVICE_RENAMES)
    return df


def pull(include_current_month: bool = False, save: bool = True) -> pd.DataFrame:
    session, token = sso_login()
    df = fetch_year(rollup_session(session, token), BASIS_YEAR, include_current_month)
    if save:
        df.to_csv(CACHE_FILE, index=False)
        print(f'Saved {len(df):,} rows → {CACHE_FILE}')
    return df


def reconcile_with_monthly_files(df: pd.DataFrame) -> pd.DataFrame:
    """Compare monthly totals with the Monthly_Cost_For_*.csv extracts FinOps.ipynb produces (same payer account)."""
    old = fr.load_cost_data(HERE)
    old = old[old['Date'].dt.year == BASIS_YEAR]
    a = df.groupby('Date')['Cost'].sum()
    b = old.groupby('Date')['Cost'].sum()
    out = pd.DataFrame({'By account (this pull)': a, 'Monthly_Cost_For_*.csv': b})
    out['Difference'] = out.iloc[:, 0] - out.iloc[:, 1]
    out['Difference %'] = out['Difference'] / out.iloc[:, 1]
    return out


# --------------------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------------------
@dataclass
class AWSForecast:
    df: pd.DataFrame
    basis_start: pd.Timestamp
    basis_end: pd.Timestamp
    horizon: pd.DatetimeIndex
    budget: float
    svc_actual: pd.Series
    mkt_actual: pd.Series
    tax_actual: pd.Series
    svc: pd.DataFrame | None = None                 # horizon × [Low, Base, High]
    params: dict = field(default_factory=dict)
    normalization: pd.Series | None = None
    account_lines: list = field(default_factory=list)
    accounts: pd.DataFrame | None = None
    account_services: dict = field(default_factory=dict)
    contracts: pd.DataFrame | None = None
    mkt: pd.Series | None = None
    mkt_detail: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)
    anomalies: pd.DataFrame | None = None

    @property
    def months(self) -> pd.DatetimeIndex:
        return self.horizon[self.horizon.year == FORECAST_YEAR]


def forecast_2027(df: pd.DataFrame, budget: float = MONTHLY_BUDGET, top_accounts: int = TOP_ACCOUNTS) -> AWSForecast:
    d = df[(df['Date'].dt.year == BASIS_YEAR) & ~df.get('Partial', False)].copy()
    basis_start, basis_end = pd.Timestamp(f'{BASIS_YEAR}-01-01'), d['Date'].max()
    months_idx = pd.date_range(basis_start, basis_end, freq='MS')
    horizon = pd.date_range(basis_end + relativedelta(months=1), f'{FORECAST_YEAR}-12-01', freq='MS')
    cat = d.pivot_table(index='Date', columns='Category', values='Cost', aggfunc='sum').reindex(months_idx).fillna(0)
    for c in (SVC, MKT, TAX):
        if c not in cat:
            cat[c] = 0.0
    fc = AWSForecast(d, basis_start, basis_end, horizon, budget, cat[SVC], cat[MKT], cat[TAX])

    # ---- Cloud Services, top-down (Savings Plans gap normalized for the trend only)
    sp = d[d['Service'] == fr.SP_NAME].groupby('Date')['Cost'].sum().reindex(months_idx).fillna(0)
    typical = sp[sp > 0].median() if (sp > 0).any() else 0
    gap = (typical - sp).where(sp < 0.5 * typical, 0)
    fc.normalization = gap[gap > 0]
    s = fc.svc_actual
    anchor, slope, raw_slope = s.tail(3).mean(), cf.theil_sen(s + gap), cf.theil_sen(s)
    h = np.arange(1, len(horizon) + 1)
    flat, damped, linear = np.full(len(h), anchor), anchor + slope * np.cumsum(cf.DAMPING ** h), anchor + slope * h
    fc.svc = pd.DataFrame({'Low': np.minimum.reduce([flat, damped, linear]), 'Base': damped,
                           'High': np.maximum.reduce([flat, damped, linear])}, index=horizon)
    fc.params = dict(ytd=s.sum(), months=len(s), anchor=anchor, slope=slope, raw_slope=raw_slope,
                     first_half=s.iloc[:len(s) // 2].mean(), peak=s.max(), peak_month=s.idxmax(),
                     low=s.min(), low_month=s.idxmin(), months_over=int((s > budget).sum()),
                     base_cap=anchor + slope * cf.DAMPING / (1 - cf.DAMPING))

    # ---- Per linked account (Cloud Services), backtested model per account
    acc = d[d['Category'] == SVC].pivot_table(index='Date', columns='AccountName', values='Cost', aggfunc='sum')
    acc = acc.reindex(months_idx).fillna(0)
    if fc.normalization is not None and len(fc.normalization):
        # put the SP add-back on the account(s) that pay the Savings Plans fee
        sp_acc = d[d['Service'] == fr.SP_NAME].groupby('AccountName')['Cost'].sum()
        if len(sp_acc):
            for a, share in (sp_acc / sp_acc.sum()).items():
                acc[a] = acc[a] + gap * share
    order = acc.sum().sort_values(ascending=False).index
    top, rest = list(order[:top_accounts]), list(order[top_accounts:])
    lines = [fd.forecast_line(a, acc[a], horizon) for a in top]
    if rest:
        lines.append(fd.forecast_line(f'All other accounts ({len(rest)})', acc[rest].sum(axis=1), horizon, force_model='Damped trend'))
    fc.account_lines = lines
    ids = d.groupby('AccountName')['AccountId'].first()
    rows = []
    for lf in lines:
        exit_rr = lf.actual.tail(3).mean()
        fy = lf.forecast.loc[fc.months].sum()
        rows.append({'Account': lf.name, 'Account ID': ids.get(lf.name, '—'), f'{BASIS_YEAR} Jan–{basis_end:%b}': lf.actual.sum(),
                     'Share': np.nan, 'Exit run-rate /mo': exit_rr, f'FY{FORECAST_YEAR} forecast': fy,
                     'FY range': f'{_money(lf.lo.loc[fc.months].sum())}–{_money(lf.hi.loc[fc.months].sum())}',
                     'Δ vs exit × 12': fy - exit_rr * 12,
                     'Model': lf.model + (' (capped)' if lf.capped else '') + (' (floored)' if lf.floored else ''),
                     'Confidence': f'{lf.confidence} (±{lf.mape:.0%})' if np.isfinite(lf.mape) else lf.confidence})
    t = pd.DataFrame(rows).set_index('Account')
    ytd_col = f'{BASIS_YEAR} Jan–{basis_end:%b}'
    t['Share'] = t[ytd_col] / t[ytd_col].sum()
    fc.accounts = t

    # top services inside each top account
    for a in top:
        sv = d[(d['Category'] == SVC) & (d['AccountName'] == a)].pivot_table(index='Date', columns='Service', values='Cost', aggfunc='sum')
        sv = sv.reindex(months_idx).fillna(0)
        sv = sv[sv.sum().sort_values(ascending=False).index[:8]]
        r = []
        for svc_name in sv.columns:
            if sv[svc_name].sum() < 100:
                continue
            lf = fd.forecast_line(svc_name, sv[svc_name], horizon)
            r.append({'Service': fr._short(svc_name, 40), ytd_col: sv[svc_name].sum(), 'Exit run-rate /mo': sv[svc_name].tail(3).mean(),
                      f'FY{FORECAST_YEAR} forecast': lf.forecast.loc[fc.months].sum(), 'Model': lf.model, 'Confidence': lf.confidence})
        fc.account_services[a] = pd.DataFrame(r).set_index('Service') if r else pd.DataFrame()

    # ---- Marketplace: (account, product) contract schedules
    m = d[d['Category'] == MKT].copy()
    m['Contract'] = [cf._contract_key('AWS', x) for x in m['Service']]
    piv = m.pivot_table(index='Date', columns=['AccountName', 'Contract'], values='Cost', aggfunc='sum').reindex(months_idx).fillna(0)
    crow, det = [], {}
    for (a, c), ser in piv.items():
        res = cf.contract_schedule(ser, horizon)
        if res is None:
            continue
        cadence, amount, sched = res
        det[(a, c)] = sched
        fy = sched[sched.index.year == FORECAST_YEAR]
        crow.append({'Contract': fr.vendor_label(c) if 'Claude' not in c else c, 'Product': c, 'Account': a, 'Cadence': cadence,
                     'Amount per charge': amount, f'{BASIS_YEAR} billed': ser.sum(),
                     f'Rest of {BASIS_YEAR} expected': sched[sched.index.year == BASIS_YEAR].sum(),
                     f'FY{FORECAST_YEAR} forecast': fy.sum(),
                     f'{FORECAST_YEAR} billing months': 'Every month' if cadence == 'Monthly' else (', '.join(fy[fy > 0].index.strftime('%b')) or '—')})
    fc.contracts = (pd.DataFrame(crow).sort_values(f'FY{FORECAST_YEAR} forecast', ascending=False).reset_index(drop=True)
                    if crow else pd.DataFrame())
    fc.mkt_detail = pd.DataFrame(det) if det else pd.DataFrame(index=horizon)
    fc.mkt = fc.mkt_detail.sum(axis=1).reindex(horizon, fill_value=0)

    fc.anomalies = find_anomalies(d, months_idx)
    _summarize(fc)
    return fc


def find_anomalies(d: pd.DataFrame, months_idx, min_typical: float = 3_000) -> pd.DataFrame:
    """Account × service lines that normally cost ≥ min_typical/month but drop to under 10% of that in some months
    (credits, billing adjustments, outages, migrations). These months distort trends, so they are listed for review."""
    x = d[d['Category'] == SVC].pivot_table(index='Date', columns=['AccountName', 'Service'], values='Cost', aggfunc='sum')
    x = x.reindex(months_idx).fillna(0)
    rows = []
    for (a, sv), ser in x.items():
        typical = ser.median()
        if typical < min_typical:
            continue
        low = ser[ser < 0.1 * typical]
        if len(low):
            rows.append({'Account': a, 'Service': fr._short(sv, 36), 'Typical /mo': typical,
                         'Months near $0': ', '.join(low.index.strftime('%b')), 'Missing vs typical': (typical - low).sum()})
    return pd.DataFrame(rows).sort_values('Missing vs typical', ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def _summarize(fc: AWSForecast) -> None:
    S, p, idx = fc.summary, fc.params, fc.months
    fy = {sc: fc.svc.loc[idx, sc].sum() for sc in ('Low', 'Base', 'High')}
    bud = fc.budget * 12
    mkt_fy = fc.mkt.loc[idx].sum()
    tax_rate = fc.tax_actual.sum() / (fc.svc_actual.sum() + fc.mkt_actual.sum())
    bottom_up = fc.accounts[f'FY{FORECAST_YEAR} forecast'].sum()
    S.update(fy=fy, budget=bud, mkt=mkt_fy, tax=(fy['Base'] + mkt_fy) * tax_rate, tax_rate=tax_rate, bottom_up=bottom_up,
             combined={k: v + mkt_fy for k, v in fy.items()})
    over = fc.svc.loc[idx, 'Base'] > fc.budget
    S['first_over'] = over.idxmax() if over.any() else None
    mn = f'{fc.basis_start:%b}–{fc.basis_end:%b %Y}'
    acc = fc.accounts[~fc.accounts.index.str.startswith('All other')]
    ytd_col = f'{BASIS_YEAR} Jan–{fc.basis_end:%b}'
    n_acc = fc.df.loc[fc.df['Category'] == SVC, 'AccountId'].nunique()
    paras = []

    gap = fc.budget - p['anchor']
    paras.append(('Where we start', [
        f'AWS Cloud Services totaled <b>{_money(p["ytd"])}</b> for {mn} across <b>{n_acc} linked accounts</b> '
        f'(pro-rated budget {_money(fc.budget * p["months"])} → {_money(fc.budget * p["months"] - p["ytd"])} '
        f'{"under" if p["ytd"] <= fc.budget * p["months"] else "over"}). Marketplace added {_money(fc.mkt_actual.sum())} '
        f'and tax {_money(fc.tax_actual.sum())}.',
        f'Exit run-rate (last 3 months) is <b>{_money(p["anchor"])}/month</b> — {p["anchor"] / fc.budget:.0%} of the '
        f'{_money(fc.budget)} monthly budget, ' + (f'{_money(gap)}/month of headroom.' if gap >= 0 else
                                                     f'<span class="bad">{_money(-gap)}/month over</span>.'),
        f'Concentration: the top 3 accounts are {acc["Share"].head(3).sum():.0%} of {BASIS_YEAR} Cloud Services — '
        + ', '.join(f'{html.escape(a)} {_money(r[ytd_col])} ({r["Share"]:.0%})' for a, r in acc.head(3).iterrows()) + '.',
    ]))

    lines = [f'Cloud Services grew about <b>{_money(p["slope"])} per month</b> in {BASIS_YEAR} '
             f'(first-half average {_money(p["first_half"])} → exit {_money(p["anchor"])}). '
             f'Range {_money(p["low"])} ({p["low_month"]:%b}) to {_money(p["peak"])} ({p["peak_month"]:%b}).']
    if fc.normalization is not None and len(fc.normalization):
        lines.append(f'Normalization: the Savings Plans fee was missing or partial in {", ".join(fc.normalization.index.strftime("%b"))} '
                     f'(≈ {_money(fc.normalization.sum())}); the typical fee is added back for the trend only, which changes the growth '
                     f'estimate from {_money(p["raw_slope"])} to {_money(p["slope"])} per month.')
    g = acc.assign(delta=acc['Δ vs exit × 12']).sort_values('delta')
    up, down = g[g['delta'] > 5_000].tail(3)[::-1], g[g['delta'] < -5_000].head(3)
    if len(up):
        def driver(a):
            t = fc.account_services.get(a)
            if t is None or t.empty:
                return ''
            g2 = d_svc[d_svc['AccountName'] == a].pivot_table(index='Date', columns='Service', values='Cost', aggfunc='sum').fillna(0)
            grow = (g2.tail(3).mean() - g2.head(3).mean()).sort_values(ascending=False).head(2)
            return '; driven by ' + ', '.join(f'{fr._short(k, 22)} {_money(g2[k].head(3).mean())}→{_money(g2[k].tail(3).mean())}/mo'
                                              for k in grow.index if grow[k] > 1_000)
        d_svc = fc.df[fc.df['Category'] == SVC]
        lines.append('Accounts adding the most in ' + str(FORECAST_YEAR) + ' (vs run-rate × 12): ' +
                     '; '.join(f'<b>{html.escape(a)}</b> +{_money(r["delta"])} ({r["Model"].lower()}, confidence {r["Confidence"]}{driver(a)})'
                               for a, r in up.iterrows()) + '.')
    if len(down):
        lines.append('Accounts trending down: ' + ', '.join(f'{html.escape(a)} {_money(r["delta"])}' for a, r in down.iterrows()) + '.')
    paras.append((f'{BASIS_YEAR} pattern', lines))

    var = bud - fy['Base']
    fo = S['first_over']
    paras.append((f'FY{FORECAST_YEAR} Cloud Services', [
        f'<b>Base {_money(fy["Base"])}</b> (low {_money(fy["Low"])} – high {_money(fy["High"])}) vs {_money(bud)} budget → '
        f'<span class="{"good" if var >= 0 else "bad"}">{_money(abs(var))} {"under" if var >= 0 else "over"}</span>. '
        f'Base run-rate reaches {_money(fc.svc["Base"].iloc[-1])}/month by Dec {FORECAST_YEAR} (ceiling ≈ {_money(p["base_cap"])})'
        + (f'; it crosses the monthly budget in {fo:%b %Y}.' if fo is not None else '; it stays within the monthly budget all year.'),
        f'Bottom-up check: the {len(fc.account_lines)} account forecasts sum to {_money(bottom_up)} '
        f'({bottom_up / fy["Base"] - 1:+.1%} vs the top-down base).',
    ]))

    c = fc.contracts
    fycol = f'FY{FORECAST_YEAR} forecast'
    mk_lines = [f'Marketplace FY{FORECAST_YEAR}: <b>{_money(mkt_fy)}</b> across {int((c[fycol] > 0).sum()) if len(c) else 0} '
                f'contracts (vs {_money(fc.mkt_actual.sum())} billed {mn}).']
    for cad in ['Quarterly', 'Annual', 'Monthly', 'Semiannual', 'Bimonthly']:
        sub = c[(c['Cadence'] == cad) & (c[fycol] > 0)] if len(c) else c
        if len(sub):
            mk_lines.append(f'{cad} ({_money(sub[fycol].sum())}): ' + ', '.join(
                f'{html.escape(fr._short(r.Contract, 28))} {_money(r[fycol])} <span class="note">[{html.escape(r.Account)}]</span>'
                for _, r in sub.iterrows()) + '.')
    q = fc.mkt.loc[idx].groupby(idx.quarter).sum()
    if len(q):
        mk_lines.append('By quarter: ' + ', '.join(f'Q{i} {_money(v)}' for i, v in q.items()) + f' — heaviest is Q{q.idxmax()}.')
    paras.append(('Marketplace', mk_lines))

    paras.append(('Total AWS funding need', [
        f'Services (base) + Marketplace = <b>{_money(S["combined"]["Base"])}</b> '
        f'(range {_money(S["combined"]["Low"])} – {_money(S["combined"]["High"])}); estimated tax at the {BASIS_YEAR} rate '
        f'({tax_rate:.1%}) ≈ {_money(S["tax"])}, all-in ≈ <b>{_money(S["combined"]["Base"] + S["tax"])}</b>.',
    ]))
    an = fc.anomalies
    an_line = ([f'Data anomalies: {len(an)} account × service lines drop to near $0 in some months (≈ {_money(an["Missing vs typical"].sum())} '
                f'below typical), e.g. ' + ', '.join(f'{html.escape(r.Account)} {html.escape(r.Service)} ({r._4})'
                                                       for r in an.head(4).itertuples()) +
                '. Short gaps (1–2 months) usually mean credits or billing adjustments and make 2026 growth look steeper than usage; '
                'gaps through the latest month usually mean a shutdown or migration — confirm both with the account owners.']
               if an is not None and len(an) else [])
    paras.append(('Assumptions & risks', an_line + [
        f'Only {p["months"]} months of {BASIS_YEAR} — seasonality cannot be estimated; Marketplace contracts billed only in Q4 '
        f'(none visible in {BASIS_YEAR} yet) are not in the numbers.',
        'Annual Marketplace lines are assumed to renew at the same price; one-off purchases in 2026 are included unless removed.',
        'New accounts or workloads launched in 2027 are not in the bottom-up view; the top-down base carries the trend.',
    ]))
    if abs(var) <= 0.01 * bud:
        rec = (f'Base lands essentially on budget ({_money(fy["Base"])} vs {_money(bud)}, {-var / bud:+.1%}) — but it crosses '
               f'{_money(fc.budget)}/month in {fo:%b %Y} and keeps rising, so H2 {FORECAST_YEAR} needs ≈ '
               f'{_money(max(fc.svc.loc[idx, "Base"].iloc[6:].mean() - fc.budget, 0))}/month of savings or growth control; '
               f'the high scenario would need {_money(fy["High"] / 12)}/month.' if fo is not None else
               f'Base lands essentially on budget ({-var / bud:+.1%}).')
    elif var > 0:
        rec = (f'Base fits the {_money(fc.budget)}/month budget with {_money(var)} to spare; the high scenario needs '
               f'{_money(fy["High"] / 12)}/month.')
    else:
        rec = (f'To hold {_money(fc.budget)}/month, Services need ≈ {_money(-var / 12)}/month of savings vs base, or a budget of '
               f'≈ {_money(fy["Base"] / 12)}/month.')
    paras.append(('Recommendation', [rec, f'Plan Marketplace separately at ≈ {_money(mkt_fy)} with the contract calendar below.']))
    S['paragraphs'] = paras


# --------------------------------------------------------------------------------------
# Tables & charts
# --------------------------------------------------------------------------------------
def monthly_table(fc: AWSForecast) -> pd.DataFrame:
    idx = fc.months
    t = pd.DataFrame({'Services — Low': fc.svc.loc[idx, 'Low'], 'Services — Base': fc.svc.loc[idx, 'Base'],
                      'Services — High': fc.svc.loc[idx, 'High'], 'Budget': fc.budget}, index=idx)
    t['Variance (Base)'] = t['Budget'] - t['Services — Base']
    t['Marketplace'] = fc.mkt.loc[idx]
    t['Combined (Base)'] = t['Services — Base'] + t['Marketplace']
    return t


def quarterly_table(fc: AWSForecast) -> pd.DataFrame:
    m = monthly_table(fc)
    q = m.groupby(m.index.quarter).sum()
    q.index = [f'Q{i} {FORECAST_YEAR}' for i in q.index]
    q.loc[f'FY{FORECAST_YEAR}'] = q.sum()
    return q


def chart_services(fc: AWSForecast) -> go.Figure:
    f, act = fc.svc, fc.svc_actual
    fig = go.Figure()
    fig.add_scatter(x=list(f.index) + list(f.index[::-1]), y=list(f['High']) + list(f['Low'][::-1]), fill='toself',
                    fillcolor=cf._rgba(C_SVC, 0.18), line=dict(width=0), name='Low–High range', hoverinfo='skip')
    fig.add_scatter(x=act.index, y=act, name=f'{BASIS_YEAR} actual', mode='lines+markers', line=dict(color=C_SVC, width=2),
                    marker=dict(size=7), hovertemplate='%{y:$,.0f}')
    fig.add_scatter(x=[act.index[-1]] + list(f.index), y=[act.iloc[-1]] + list(f['Base']), name='Base forecast',
                    mode='lines+markers', line=dict(color=C_SVC, width=2, dash='dash'), marker=dict(size=5), hovertemplate='%{y:$,.0f}')
    fig.add_hline(y=fc.budget, line=dict(color=fr.INK, width=1.5),
                  annotation=dict(text=f'Budget {_money(fc.budget)}/mo', font=dict(color=fr.INK, size=11)), annotation_position='top left')
    fig.add_vline(x=fc.horizon[0] - pd.Timedelta(days=15), line=dict(color=fr.AXIS, width=1))
    fig.add_annotation(x=fc.horizon[0], y=1, yref='paper', text='Forecast →', showarrow=False, xanchor='left',
                       font=dict(color=fr.MUTED, size=11))
    fig = _layout(fig, 380)
    fig.update_yaxes(range=[0, max(f['High'].max(), fc.budget, act.max()) * 1.08])
    return fig


def chart_marketplace(fc: AWSForecast) -> go.Figure:
    det = fc.mkt_detail
    names = [', '.join(fr._short(c, 22) for (a, c), v in det.loc[d].items() if v > 1000) if len(det.columns) else ''
             for d in fc.mkt.index]
    fig = go.Figure()
    fig.add_bar(x=fc.mkt_actual.index, y=fc.mkt_actual, name=f'{BASIS_YEAR} actual', marker=dict(color=C_MKT, line=dict(width=0)),
                hovertemplate='%{y:$,.0f}')
    fig.add_bar(x=fc.mkt.index, y=fc.mkt, name='Forecast', customdata=names,
                marker=dict(color=cf._rgba(C_MKT, 0.45), line=dict(color=C_MKT, width=1)),
                hovertemplate='%{y:$,.0f}<br>%{customdata}<extra></extra>')
    fig.add_vline(x=fc.horizon[0] - pd.Timedelta(days=15), line=dict(color=fr.AXIS, width=1))
    return _layout(fig, 340)


def chart_combined(fc: AWSForecast) -> go.Figure:
    fig = go.Figure()
    for name, a, f, col in [('Cloud Services', fc.svc_actual, fc.svc['Base'], C_SVC), ('Marketplace', fc.mkt_actual, fc.mkt, C_MKT)]:
        fig.add_bar(x=list(a.index) + list(f.index), y=list(a) + list(f), name=name,
                    marker=dict(color=[col] * len(a) + [cf._rgba(col, 0.5)] * len(f), line=dict(width=0)), hovertemplate='%{y:$,.0f}')
    fig.update_layout(barmode='stack')
    fig.add_hline(y=fc.budget, line=dict(color=fr.INK, width=1.5),
                  annotation=dict(text=f'Services budget {_money(fc.budget)}/mo', font=dict(color=fr.INK, size=11)),
                  annotation_position='top left')
    fig.add_vline(x=fc.horizon[0] - pd.Timedelta(days=15), line=dict(color=fr.AXIS, width=1))
    return _layout(fig, 380)


def chart_accounts_share(fc: AWSForecast) -> go.Figure:
    t = fc.accounts.iloc[::-1]
    col = f'FY{FORECAST_YEAR} forecast'
    fig = go.Figure(go.Bar(y=[fr._short(a, 34) for a in t.index], x=t[col], orientation='h',
                           marker=dict(color=C_SVC, line=dict(width=0)), text=[_money(v) for v in t[col]],
                           textposition='outside', textfont=dict(color=fr.INK2, size=11),
                           hovertemplate='%{y}: %{x:$,.0f}<extra></extra>'))
    fig = _layout(fig, 28 * len(t) + 60, legend=False)
    fig.update_layout(hovermode='closest')
    fig.update_xaxes(tickformat='$.2~s', showgrid=True, gridcolor=fr.GRID, range=[0, t[col].max() * 1.2])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=fr.INK2))
    return fig


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _anomaly_html(fc: AWSForecast) -> str:
    an = fc.anomalies
    if an is None or not len(an):
        return ''
    t = an.set_index('Account')
    return (f'<section><h2>Data anomalies to confirm</h2><p class="lede">Lines that normally cost ≥ $3K/month but show near $0 in some '
            f'months. Short gaps are usually credits or billing adjustments; gaps through the latest month usually mean a shutdown or '
            f'migration. Both distort the 2026 trend, so confirm them before locking the budget.</p>'
            f'{fr._table(t, {"Typical /mo": _money, "Missing vs typical": _money})}</section>')


def build_report(fc: AWSForecast, out_path: str | None = None, recon: pd.DataFrame | None = None) -> str:
    S, p = fc.summary, fc.params
    fy = f'FY{FORECAST_YEAR}'
    out_path = out_path or os.path.join(HERE, f'AWS_Forecast_By_Account_{FORECAST_YEAR}.html')

    def tile(k, v, d):
        return f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'

    var = S['budget'] - S['fy']['Base']
    tiles = '<div class="tiles">' + ''.join([
        tile(f'{BASIS_YEAR} Cloud Services ({fc.basis_start:%b}–{fc.basis_end:%b})', _money(p['ytd']),
             f'{fc.df.loc[fc.df["Category"] == SVC, "AccountId"].nunique()} linked accounts'),
        tile('Exit run-rate', _money(p['anchor']) + '/mo', f'{p["anchor"] / fc.budget:.0%} of {_money(fc.budget)} budget'),
        tile(f'{fy} Services · base', _money(S['fy']['Base']),
             f'<span class="{"good" if var >= 0 else "bad"}">{_money(abs(var))} {"under" if var >= 0 else "over"}</span> {_money(S["budget"])}'),
        tile(f'{fy} Marketplace', _money(S['mkt']), f'{BASIS_YEAR} billed {_money(fc.mkt_actual.sum())}'),
        tile(f'{fy} Combined · base', _money(S['combined']['Base']),
             f'range {_money(S["combined"]["Low"])}–{_money(S["combined"]["High"])}'),
        tile('All-in incl. est. tax', _money(S['combined']['Base'] + S['tax']), f'tax ≈ {_money(S["tax"])}'),
    ]) + '</div>'
    summary = ''.join(f'<h3>{html.escape(t)}</h3><ul class="ins a">' + ''.join(f'<li>{l}</li>' for l in ls) + '</ul>'
                      for t, ls in S['paragraphs'])

    acc = fc.accounts.drop(columns=['FY range']).copy()     # ranges are shown in the account trend charts
    ytd_col = f'{BASIS_YEAR} Jan–{fc.basis_end:%b}'
    acc_html = fr._table(acc, {ytd_col: _money, 'Share': lambda v: f'{v:.0%}', 'Exit run-rate /mo': _money,
                               f'{fy} forecast': _money, 'Δ vs exit × 12': lambda v: '$0' if abs(v) < 500 else ('+' if v > 0 else '') + _money(v)})
    drill = ''
    for a, t in fc.account_services.items():
        if t.empty:
            continue
        drill += (f'<details><summary>{html.escape(a)} — top services</summary>'
                  f'{fr._table(t, {ytd_col: _money, "Exit run-rate /mo": _money, f"{fy} forecast": _money})}</details>')

    c = fc.contracts.copy()
    if len(c):
        c = c.set_index('Contract').drop(columns=['Product'])
        mk_html = fr._table(c, {'Amount per charge': _money, f'{BASIS_YEAR} billed': _money,
                                f'Rest of {BASIS_YEAR} expected': _money, f'{fy} forecast': _money})
    else:
        mk_html = '<p class="note">No Marketplace charges in the basis year.</p>'

    mt = monthly_table(fc)
    mt.index = mt.index.strftime('%b %Y')
    recon_html = ''
    if recon is not None and len(recon):
        r = recon.copy()
        r.index = r.index.strftime('%b %Y')
        recon_html = (f'<details><summary>Data check — this pull vs Monthly_Cost_For_*.csv</summary>'
                      f'{fr._table(r, {r.columns[0]: _money, r.columns[1]: _money, "Difference": _money, "Difference %": lambda v: f"{v:+.2%}"})}'
                      f'</details>')

    sm = SimpleNamespace(lines={'acct': fc.account_lines})
    figs = [chart_services(fc), chart_accounts_share(fc), fd.chart_small_multiples(sm, 'acct', C_SVC), chart_marketplace(fc), chart_combined(fc)]
    fh = [fr._fig_html(f, i == 0) for i, f in enumerate(figs)]

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AWS Forecast by Account</title><style>{fr.css()}</style></head>
<body><main>
<header><div><h1>AWS {fy} Forecast — by Linked Account</h1>
<p class="sub">Roll-up account {html.escape(ROLLUP_ACCOUNT_NAME)} ({ROLLUP_ACCOUNT_ID}) · basis {fc.basis_start:%b}–{fc.basis_end:%b %Y} only</p></div>
<div class="meta">Cost Explorer · {METRIC} · Services budget {_money(fc.budget)}/mo<br>Generated {datetime.now():%Y-%m-%d %H:%M}</div></header>
{tiles}
<section><h2>Executive summary</h2>{summary}
<details><summary>How the forecast is built</summary><ul class="ins a">
<li><b>Data:</b> one Cost Explorer pull from the roll-up account grouped by <code>LINKED_ACCOUNT</code> × <code>SERVICE</code>;
AWS vs Marketplace from <code>BILLING_ENTITY</code>.</li>
<li><b>Services (top-down):</b> start at the last-3-month average; trend = Theil-Sen slope of {BASIS_YEAR}. Low = flat, Base = damped trend
(momentum fades {1 - cf.DAMPING:.0%}/month), High = trend continues.</li>
<li><b>Services (by account):</b> each account picks the model (flat / damped / linear / Holt ETS) that best re-predicted its own
{BASIS_YEAR} months; kept within 50%–2× of its run-rate. Accounts are summed as an independent check.</li>
<li><b>Marketplace:</b> each account × product repeats its {BASIS_YEAR} billing cadence and last amount.</li></ul></details>
{recon_html}</section>

<section><h2>a) Cloud Services — {fy} forecast</h2>{fh[0]}
<h3>By quarter</h3>{cf._ftable(quarterly_table(fc), 'Quarter', var_cols=('Variance',))}
<details><summary>By month</summary>{cf._ftable(mt, 'Month', var_cols=('Variance',))}</details></section>

<section><h2>Cloud Services by linked account</h2>
<p class="lede">{fy} base forecast per account. Bottom-up total {_money(S['bottom_up'])} vs top-down {_money(S['fy']['Base'])}
({S['bottom_up'] / S['fy']['Base'] - 1:+.1%}).</p>{fh[1]}{acc_html}
<h3>Account trends</h3>{fh[2]}
<h3>Top services inside each account</h3>{drill}</section>

{_anomaly_html(fc)}
<section><h2>b) Marketplace — {fy} forecast</h2>{fh[3]}<h3>Contract schedule (by account)</h3>{mk_html}</section>

<section><h2>c) Combined — Services + Marketplace</h2>{fh[4]}</section>
<p class="note">Source: AWS Cost Explorer via {html.escape(ROLLUP_ACCOUNT_NAME)} · cached in {os.path.basename(CACHE_FILE)} · logic: AWS_FinOps_By_Account.py</p>
</main></body></html>"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cached', action='store_true', help=f'use {os.path.basename(CACHE_FILE)} instead of calling AWS')
    ap.add_argument('--budget', type=float, default=MONTHLY_BUDGET)
    ap.add_argument('--theme', choices=list(fr.THEMES), default='dark')
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    fr.set_theme(a.theme)
    df = load_cached() if a.cached else pull()
    recon = reconcile_with_monthly_files(df)
    fc = forecast_2027(df, a.budget)
    print('Report written:', build_report(fc, a.out, recon))
    S = fc.summary
    print(f"FY{FORECAST_YEAR} Services base {_money(S['fy']['Base'])} (low {_money(S['fy']['Low'])}, high {_money(S['fy']['High'])}) "
          f"vs budget {_money(S['budget'])} · Marketplace {_money(S['mkt'])} · bottom-up by account {_money(S['bottom_up'])}")


if __name__ == '__main__':
    main()
