"""Interceptor performance model.

Computes the maximum reach of a boost-then-coast interceptor: constant
acceleration up to burnout speed, then unpowered coast for the remainder of
the engagement window.
"""


def interceptor_range_km(
    T_s: float,
    burnout_velocity_km_s: float,
    max_accel_g: float,
    g0_m_s2: float,
) -> float:
    """
    Maximum straight-line distance (km) reachable by an interceptor within
    time T_s, assuming constant acceleration max_accel_g * g0 until the burnout
    velocity, then coasting at that speed (straight line, no gravity or flight-path model).

    Parameters
    ----------
    T_s      : engagement time window, s
    burnout_velocity_km_s : burnout velocity, km/s
    max_accel_g           : acceleration during boost, in multiples of g0
    g0_m_s2               : standard gravity, m/s^2
    """
    if T_s <= 0.0:
        return 0.0

    a = max_accel_g * g0_m_s2          # boost acceleration, m/s^2
    v = burnout_velocity_km_s * 1000.0        # burnout speed, m/s
    t_a = v / a                # time to reach burnout speed, s

    if T_s >= t_a:
        # Full boost phase plus coast: d = v^2/(2a) + (T - t_a) * v
        d_m = 0.5 * (v**2) / a + (T_s - t_a) * v
    else:
        # Window ends during boost: d = a*T^2/2
        d_m = 0.5 * a * (T_s**2)

    return d_m / 1000.0
