#!/usr/bin/env python3
"""
Example: Valuing a hypothetical company using Damodaran's DCF framework.

This script demonstrates both FCFF and FCFE approaches, plus sensitivity analysis.
"""

from stock_analysis_dcf import (
    CompanyData,
    GrowthAssumptions,
    dcf_fcff,
    dcf_fcfe,
    sensitivity_analysis,
    print_valuation_report,
    print_sensitivity_table,
)


def main():
    # --- Define company financials (hypothetical tech company) ---
    company = CompanyData(
        name="Acme Tech Corp",
        ticker="ATEC",
        revenue=10_000,          # $10B revenue
        ebit=1_500,              # $1.5B EBIT (15% margin)
        interest_expense=200,    # $200M interest
        tax_rate=0.21,           # 21% tax rate
        total_debt=4_000,        # $4B debt
        cash_and_equivalents=2_000,  # $2B cash
        shares_outstanding=500,  # 500M shares
        book_value_of_equity=8_000,
        current_stock_price=45.00,
        beta=1.15,
    )

    # --- Growth assumptions ---
    assumptions = GrowthAssumptions(
        revenue_growth_rate=0.12,       # 12% revenue growth in high-growth phase
        high_growth_years=5,            # 5-year high-growth period
        target_operating_margin=0.18,   # Margin expands to 18%
        stable_growth_rate=0.025,       # 2.5% perpetual growth
        stable_operating_margin=0.15,   # Stable margin of 15%
        risk_free_rate=0.04,            # 4% risk-free rate
        equity_risk_premium=0.055,      # 5.5% equity risk premium
        cost_of_debt_pretax=0.05,       # 5% pre-tax cost of debt
        debt_to_capital_ratio=0.25,     # 25% debt-to-capital
        sales_to_capital_ratio=2.0,     # $2 of revenue per $1 of capital
    )

    # --- Run FCFF valuation ---
    print("\n" + "=" * 65)
    print("  FCFF (Free Cash Flow to Firm) Approach")
    print("=" * 65)
    fcff_result = dcf_fcff(company, assumptions)
    print_valuation_report(fcff_result)

    # --- Run FCFE valuation ---
    print("\n" + "=" * 65)
    print("  FCFE (Free Cash Flow to Equity) Approach")
    print("=" * 65)
    fcfe_result = dcf_fcfe(company, assumptions)
    print_valuation_report(fcfe_result)

    # --- Sensitivity analysis ---
    table = sensitivity_analysis(company, assumptions)
    print_sensitivity_table(table, company.name)

    # --- Summary ---
    print(f"\n{'=' * 65}")
    print("  Valuation Summary")
    print(f"{'=' * 65}")
    print(f"  FCFF intrinsic value:  ${fcff_result.value_per_share:,.2f}/share")
    print(f"  FCFE intrinsic value:  ${fcfe_result.value_per_share:,.2f}/share")
    print(f"  Current market price:  ${company.current_stock_price:,.2f}/share")
    print(f"  FCFF implied upside:   {fcff_result.implied_upside:+.2f}%")
    print(f"  FCFE implied upside:   {fcfe_result.implied_upside:+.2f}%")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
