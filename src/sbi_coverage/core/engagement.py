def interceptor_range_km(
    T_s: float,
    v_km_s: float,
    a_g: float,
    g0_m_s2: float,
) -> float:
    """
    Maximum straight-line distance reachable by an interceptor
    within time T_s, assuming constant acceleration a_g*g0 until
    burnout speed v_km_s, then coasting.
    """
    if T_s <= 0.0:
        return 0.0

    a = a_g * g0_m_s2
    v = v_km_s * 1000.0
    t_a = v / a

    if T_s >= t_a:
        d_m = 0.5 * (v**2) / a + (T_s - t_a) * v
    else:
        d_m = 0.5 * a * (T_s**2)

    return d_m / 1000.0