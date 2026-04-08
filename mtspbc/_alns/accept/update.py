def update(current: float, step: float, method: str) -> float:
    method = method.lower()
    if method == "linear":
        return current - step
    if method == "exponential":
        return current * step
    raise ValueError(f"Method `{method}' not understood.")
