def format_salary(lowest: float, highest: float) -> str:
    """Format salary range in a human-readable format."""
    if not lowest and not highest:
        return "Salary not specified"
    if lowest == highest:
        return f"${lowest:,.0f}"
    return f"${lowest:,.0f} - ${highest:,.0f}"
