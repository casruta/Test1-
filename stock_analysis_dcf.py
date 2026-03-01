"""
Stock Valuation using Damodaran's DCF (Discounted Cash Flow) Methodology.

Implements a two-stage DCF model following Aswath Damodaran's framework:
  - Stage 1: High-growth period with explicit cash flow projections
  - Stage 2: Terminal value using a stable-growth perpetuity model

Supports both:
  - FCFF (Free Cash Flow to Firm) -> Enterprise Value via WACC
  - FCFE (Free Cash Flow to Equity) -> Equity Value via Cost of Equity
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompanyData:
    """Financial inputs for a company to be valued."""

    name: str
    ticker: str

    # Income statement
    revenue: float  # Current annual revenue
    ebit: float  # Earnings before interest and taxes
    interest_expense: float = 0.0
    tax_rate: float = 0.25  # Effective tax rate

    # Balance sheet
    total_debt: float = 0.0
    cash_and_equivalents: float = 0.0
    shares_outstanding: float = 1.0  # In millions
    book_value_of_equity: float = 0.0

    # Cash flow items
    capex: float = 0.0
    depreciation: float = 0.0
    change_in_working_capital: float = 0.0

    # Market data
    current_stock_price: float = 0.0
    beta: float = 1.0


@dataclass
class GrowthAssumptions:
    """Assumptions for the DCF projection following Damodaran's framework."""

    # High-growth phase
    revenue_growth_rate: float = 0.10  # Annual revenue growth in high-growth phase
    high_growth_years: int = 5  # Duration of high-growth phase
    target_operating_margin: float = 0.15  # Operating margin at end of high growth

    # Stable-growth phase
    stable_growth_rate: float = 0.025  # Perpetual growth rate (≤ risk-free rate)
    stable_operating_margin: float = 0.12  # Long-run operating margin

    # Cost of capital inputs
    risk_free_rate: float = 0.04  # 10-year government bond yield
    equity_risk_premium: float = 0.055  # Market risk premium
    cost_of_debt_pretax: float = 0.05  # Pre-tax cost of debt
    debt_to_capital_ratio: float = 0.20  # Target D/(D+E)

    # Reinvestment
    sales_to_capital_ratio: float = 2.0  # Revenue / Invested Capital


@dataclass
class DCFResult:
    """Output of a DCF valuation."""

    company_name: str
    ticker: str
    method: str  # "FCFF" or "FCFE"

    projected_revenues: list = field(default_factory=list)
    projected_ebit: list = field(default_factory=list)
    projected_fcff: list = field(default_factory=list)
    projected_fcfe: list = field(default_factory=list)

    pv_cash_flows: float = 0.0
    terminal_value: float = 0.0
    pv_terminal_value: float = 0.0
    enterprise_value: float = 0.0
    equity_value: float = 0.0
    value_per_share: float = 0.0

    wacc: float = 0.0
    cost_of_equity: float = 0.0
    current_price: float = 0.0
    implied_upside: float = 0.0


def cost_of_equity(beta: float, risk_free_rate: float, equity_risk_premium: float) -> float:
    """CAPM: Ke = Rf + Beta * ERP"""
    return risk_free_rate + beta * equity_risk_premium


def wacc(
    cost_of_equity_val: float,
    cost_of_debt_pretax: float,
    tax_rate: float,
    debt_to_capital: float,
) -> float:
    """Weighted Average Cost of Capital."""
    equity_weight = 1.0 - debt_to_capital
    after_tax_debt = cost_of_debt_pretax * (1 - tax_rate)
    return equity_weight * cost_of_equity_val + debt_to_capital * after_tax_debt


def dcf_fcff(company: CompanyData, assumptions: GrowthAssumptions) -> DCFResult:
    """
    Two-stage FCFF DCF following Damodaran's approach.

    Stage 1: Project revenue growth, converge operating margin toward target,
             compute FCFF = EBIT(1-t) - Reinvestment
    Stage 2: Terminal value = FCFF_stable / (WACC - g_stable)
    """
    result = DCFResult(
        company_name=company.name,
        ticker=company.ticker,
        method="FCFF",
        current_price=company.current_stock_price,
    )

    # Cost of capital
    ke = cost_of_equity(company.beta, assumptions.risk_free_rate, assumptions.equity_risk_premium)
    discount_rate = wacc(
        ke, assumptions.cost_of_debt_pretax, company.tax_rate, assumptions.debt_to_capital_ratio
    )
    result.wacc = discount_rate
    result.cost_of_equity = ke

    n = assumptions.high_growth_years
    current_revenue = company.revenue
    current_margin = company.ebit / company.revenue if company.revenue else 0.0
    target_margin = assumptions.target_operating_margin

    # Linearly converge operating margin from current to target over high-growth period
    margin_step = (target_margin - current_margin) / max(n, 1)

    pv_sum = 0.0
    revenue = current_revenue
    margin = current_margin

    for year in range(1, n + 1):
        revenue *= 1 + assumptions.revenue_growth_rate
        margin += margin_step
        ebit = revenue * margin
        nopat = ebit * (1 - company.tax_rate)

        # Reinvestment = Change in Revenue / Sales-to-Capital ratio
        delta_revenue = revenue - (
            current_revenue * (1 + assumptions.revenue_growth_rate) ** (year - 1)
            if year > 1
            else current_revenue
        )
        reinvestment = delta_revenue / assumptions.sales_to_capital_ratio
        fcff = nopat - reinvestment

        discount_factor = (1 + discount_rate) ** year
        pv_sum += fcff / discount_factor

        result.projected_revenues.append(round(revenue, 2))
        result.projected_ebit.append(round(ebit, 2))
        result.projected_fcff.append(round(fcff, 2))

    # Terminal year
    terminal_revenue = revenue * (1 + assumptions.stable_growth_rate)
    terminal_ebit = terminal_revenue * assumptions.stable_operating_margin
    terminal_nopat = terminal_ebit * (1 - company.tax_rate)
    terminal_reinvestment = (
        terminal_revenue - revenue
    ) / assumptions.sales_to_capital_ratio
    terminal_fcff = terminal_nopat - terminal_reinvestment

    terminal_value = terminal_fcff / (discount_rate - assumptions.stable_growth_rate)
    pv_terminal = terminal_value / (1 + discount_rate) ** n

    result.pv_cash_flows = round(pv_sum, 2)
    result.terminal_value = round(terminal_value, 2)
    result.pv_terminal_value = round(pv_terminal, 2)

    enterprise_value = pv_sum + pv_terminal
    equity_value = enterprise_value - company.total_debt + company.cash_and_equivalents

    result.enterprise_value = round(enterprise_value, 2)
    result.equity_value = round(equity_value, 2)
    result.value_per_share = round(equity_value / company.shares_outstanding, 2)

    if company.current_stock_price > 0:
        result.implied_upside = round(
            (result.value_per_share / company.current_stock_price - 1) * 100, 2
        )

    return result


def dcf_fcfe(company: CompanyData, assumptions: GrowthAssumptions) -> DCFResult:
    """
    Two-stage FCFE DCF (discount at cost of equity instead of WACC).

    FCFE = Net Income - Reinvestment + Net Borrowing
    """
    result = DCFResult(
        company_name=company.name,
        ticker=company.ticker,
        method="FCFE",
        current_price=company.current_stock_price,
    )

    ke = cost_of_equity(company.beta, assumptions.risk_free_rate, assumptions.equity_risk_premium)
    result.cost_of_equity = ke
    result.wacc = wacc(
        ke, assumptions.cost_of_debt_pretax, company.tax_rate, assumptions.debt_to_capital_ratio
    )

    n = assumptions.high_growth_years
    current_revenue = company.revenue
    current_margin = company.ebit / company.revenue if company.revenue else 0.0
    target_margin = assumptions.target_operating_margin
    margin_step = (target_margin - current_margin) / max(n, 1)

    pv_sum = 0.0
    revenue = current_revenue
    margin = current_margin

    for year in range(1, n + 1):
        revenue *= 1 + assumptions.revenue_growth_rate
        margin += margin_step
        ebit = revenue * margin
        net_income = (ebit - company.interest_expense) * (1 - company.tax_rate)

        delta_revenue = revenue - (
            current_revenue * (1 + assumptions.revenue_growth_rate) ** (year - 1)
            if year > 1
            else current_revenue
        )
        reinvestment = delta_revenue / assumptions.sales_to_capital_ratio
        # Net borrowing = debt ratio * reinvestment
        net_borrowing = assumptions.debt_to_capital_ratio * reinvestment
        fcfe = net_income - reinvestment + net_borrowing

        discount_factor = (1 + ke) ** year
        pv_sum += fcfe / discount_factor

        result.projected_revenues.append(round(revenue, 2))
        result.projected_ebit.append(round(ebit, 2))
        result.projected_fcfe.append(round(fcfe, 2))

    # Terminal value (equity)
    terminal_revenue = revenue * (1 + assumptions.stable_growth_rate)
    terminal_ebit = terminal_revenue * assumptions.stable_operating_margin
    terminal_net_income = (terminal_ebit - company.interest_expense) * (1 - company.tax_rate)
    terminal_reinvestment = (terminal_revenue - revenue) / assumptions.sales_to_capital_ratio
    terminal_net_borrowing = assumptions.debt_to_capital_ratio * terminal_reinvestment
    terminal_fcfe = terminal_net_income - terminal_reinvestment + terminal_net_borrowing

    terminal_value = terminal_fcfe / (ke - assumptions.stable_growth_rate)
    pv_terminal = terminal_value / (1 + ke) ** n

    result.pv_cash_flows = round(pv_sum, 2)
    result.terminal_value = round(terminal_value, 2)
    result.pv_terminal_value = round(pv_terminal, 2)

    equity_value = pv_sum + pv_terminal + company.cash_and_equivalents
    result.equity_value = round(equity_value, 2)
    result.enterprise_value = round(equity_value + company.total_debt - company.cash_and_equivalents, 2)
    result.value_per_share = round(equity_value / company.shares_outstanding, 2)

    if company.current_stock_price > 0:
        result.implied_upside = round(
            (result.value_per_share / company.current_stock_price - 1) * 100, 2
        )

    return result


def sensitivity_analysis(
    company: CompanyData,
    assumptions: GrowthAssumptions,
    growth_range: Optional[list] = None,
    wacc_range: Optional[list] = None,
) -> dict:
    """
    Run sensitivity analysis varying revenue growth and WACC.

    Returns a dict mapping (growth_rate, wacc_adjustment) -> value_per_share.
    """
    if growth_range is None:
        base_g = assumptions.revenue_growth_rate
        growth_range = [base_g - 0.04, base_g - 0.02, base_g, base_g + 0.02, base_g + 0.04]
    if wacc_range is None:
        wacc_range = [-0.02, -0.01, 0.0, 0.01, 0.02]

    results = {}
    for g in growth_range:
        for w_adj in wacc_range:
            modified = GrowthAssumptions(
                revenue_growth_rate=g,
                high_growth_years=assumptions.high_growth_years,
                target_operating_margin=assumptions.target_operating_margin,
                stable_growth_rate=assumptions.stable_growth_rate,
                stable_operating_margin=assumptions.stable_operating_margin,
                risk_free_rate=assumptions.risk_free_rate + w_adj,
                equity_risk_premium=assumptions.equity_risk_premium,
                cost_of_debt_pretax=assumptions.cost_of_debt_pretax + w_adj,
                debt_to_capital_ratio=assumptions.debt_to_capital_ratio,
                sales_to_capital_ratio=assumptions.sales_to_capital_ratio,
            )
            r = dcf_fcff(company, modified)
            results[(round(g * 100, 1), round(w_adj * 100, 1))] = r.value_per_share
    return results


def print_valuation_report(result: DCFResult) -> None:
    """Print a formatted valuation summary."""
    print("=" * 65)
    print(f"  DCF Valuation Report: {result.company_name} ({result.ticker})")
    print(f"  Method: {result.method}")
    print("=" * 65)

    print(f"\n  WACC:             {result.wacc:.2%}")
    print(f"  Cost of Equity:   {result.cost_of_equity:.2%}")

    print(f"\n  {'Year':<6} {'Revenue':>14} {'EBIT':>14} {'FCF':>14}")
    print("  " + "-" * 50)
    fcf_list = result.projected_fcff if result.method == "FCFF" else result.projected_fcfe
    for i, (rev, ebit, fcf) in enumerate(
        zip(result.projected_revenues, result.projected_ebit, fcf_list), 1
    ):
        print(f"  {i:<6} {rev:>14,.2f} {ebit:>14,.2f} {fcf:>14,.2f}")

    print(f"\n  PV of Cash Flows:     {result.pv_cash_flows:>14,.2f}")
    print(f"  Terminal Value:       {result.terminal_value:>14,.2f}")
    print(f"  PV of Terminal Value: {result.pv_terminal_value:>14,.2f}")
    print(f"  Enterprise Value:     {result.enterprise_value:>14,.2f}")
    print(f"  Equity Value:         {result.equity_value:>14,.2f}")
    print(f"\n  Intrinsic Value/Share:  ${result.value_per_share:,.2f}")
    if result.current_price > 0:
        print(f"  Current Market Price:   ${result.current_price:,.2f}")
        print(f"  Implied Upside:         {result.implied_upside:+.2f}%")
    print("=" * 65)


def print_sensitivity_table(table: dict, company_name: str) -> None:
    """Print a formatted sensitivity analysis table."""
    # Extract unique growth rates and WACC adjustments
    growth_rates = sorted(set(k[0] for k in table))
    wacc_adjs = sorted(set(k[1] for k in table))

    print(f"\n{'=' * 65}")
    print(f"  Sensitivity Analysis: {company_name}")
    print(f"  Revenue Growth (%) vs WACC Adjustment (%)")
    print(f"{'=' * 65}")

    header = f"  {'Growth':>8}"
    for w in wacc_adjs:
        header += f"  {w:>+6.1f}%"
    print(header)
    print("  " + "-" * (10 + len(wacc_adjs) * 9))

    for g in growth_rates:
        row = f"  {g:>7.1f}%"
        for w in wacc_adjs:
            val = table.get((g, w), 0)
            row += f"  ${val:>6.0f}"
        print(row)
    print(f"{'=' * 65}")
