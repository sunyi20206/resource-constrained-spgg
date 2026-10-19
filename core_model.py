import numpy as np
from numba import jit

# Center followed by the four nearest neighbors.
VN_OFFSETS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)

# Thirteen sites with Manhattan distance at most two, including the center.
EXTENDED_OFFSETS = np.array([
    [0, 0],
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-2, 0], [2, 0], [0, -2], [0, 2],
    [-1, -1], [-1, 1], [1, -1], [1, 1]
], dtype=np.int64)

@jit(nopython=True, cache=True)
def calculate_payoffs_v2(S, P_accum, r_gain, c, alpha, theta, mu):
    """Return individual net payoffs and organizer commission minus organizational cost.

    S encodes cooperation as 0, defection as 1, and vacancies as 2.
    P_accum contains resource balances before the current round.
    """
    L = S.shape[0]

    p_round = np.zeros((L, L), dtype=np.float64)

    psi_round = np.zeros((L, L), dtype=np.float64)

    beneficiary_pool = np.zeros((L, L), dtype=np.float64)

    groups_joined_count = np.zeros((L, L), dtype=np.int32)

    is_active_organizer = np.zeros((L, L), dtype=np.bool_)

    for r in range(L):
        for col in range(L):

            if S[r, col] != 2 and P_accum[r, col] >= 0:
                is_active_organizer[r, col] = True

                nc_in_group = 0
                n_active_in_group = 0
                for i in range(5):
                    dr, dc = VN_OFFSETS[i]
                    nr, ncol = (r + dr) % L, (col + dc) % L
                    if S[nr, ncol] != 2:
                        n_active_in_group += 1
                        if S[nr, ncol] == 0:
                            nc_in_group += 1

                psi_round[r, col] = alpha * r_gain * c * nc_in_group - 5 * theta

                remaining_benefit = (1.0 - alpha) * r_gain * c * nc_in_group

                if n_active_in_group > 0:
                    share_per_member = remaining_benefit / n_active_in_group
                    for i in range(5):
                        dr, dc = VN_OFFSETS[i]
                        nr, ncol = (r + dr) % L, (col + dc) % L
                        if S[nr, ncol] != 2:
                            beneficiary_pool[nr, ncol] += share_per_member
                            groups_joined_count[nr, ncol] += 1

    for r in range(L):
        for col in range(L):
            if S[r, col] == 2:
                continue

            total_investment_cost = 0.0
            if S[r, col] == 0:
                total_investment_cost = groups_joined_count[r, col] * c

            p_round[r, col] = psi_round[r, col] + (beneficiary_pool[r, col] - total_investment_cost) - mu

    return p_round, psi_round

@jit(nopython=True, cache=True)
def update_strategies_fermi_v2(S, p_round, P_accum, K_fermi=0.5):
    """Update strategies synchronously using round payoffs and settled resources."""
    L = S.shape[0]
    # Read imitation sources from S and write changes only to S_next.
    S_next = np.copy(S)
    c2d, d2c = 0, 0

    is_frozen = np.zeros((L, L), dtype=np.bool_)

    for r in range(L):
        for col in range(L):
            if S[r, col] == 2: continue

            # Freeze only when every occupied site in the extended neighborhood has a negative balance.
            all_bankrupt = True
            for i in range(13):
                dr, dc = EXTENDED_OFFSETS[i]
                nr, ncol = (r + dr) % L, (col + dc) % L

                if S[nr, ncol] != 2 and P_accum[nr, ncol] >= 0:
                    all_bankrupt = False
                    break

            if all_bankrupt:

                is_frozen[r, col] = True
                continue

            # Frozen neighbors remain possible imitation sources.
            neighbor_idx = np.random.randint(1, 5)
            dr, dc = VN_OFFSETS[neighbor_idx]
            nr, ncol = (r + dr) % L, (col + dc) % L

            if S[nr, ncol] == 2: continue

            delta_p = p_round[nr, ncol] - p_round[r, col]
            prob = 1.0 / (1.0 + np.exp(-delta_p / K_fermi))

            if np.random.random() < prob:
                old_strat = S[r, col]
                new_strat = S[nr, ncol]

                if old_strat != new_strat:
                    S_next[r, col] = new_strat

                    if old_strat == 0:
                        c2d += 1
                    else:
                        d2c += 1

    return S_next, c2d, d2c, is_frozen

def run_simulation(L, steps, mc, rho, params):
    """Return averaged trajectories over mc independent realizations.

    The first four series are overall cooperation, non-frozen cooperators
    divided by population size, frozen fraction, and cooperation among
    non-frozen players.
    """

    rc_avg = np.zeros(steps)
    rho_avg = np.zeros(steps)
    c2d_avg = np.zeros(steps)
    d2c_avg = np.zeros(steps)

    fc_all_avg = np.zeros(steps)
    fc_eff_avg = np.zeros(steps)
    phi_f_avg = np.zeros(steps)
    fc_act_avg = np.zeros(steps)

    ss_fc_all_list = []
    ss_fc_eff_list = []

    curr_r = params.get('r', 0)
    curr_mu = params.get('mu', 0)
    curr_alpha = params.get('alpha', 0)
    curr_theta = params.get('theta', 0)

    for m in range(mc):

        N_max = L * L
        num_players = int(np.floor(N_max * rho))

        S = np.full((L, L), 2, dtype=np.int8)

        p_prev = np.zeros((L, L))

        indices = np.random.choice(N_max, size=num_players, replace=False)

        num_C = num_players // 2

        num_D = num_players - num_C

        strats = np.array([0] * num_C + [1] * num_D, dtype=np.int8)

        np.random.shuffle(strats)

        S.ravel()[indices] = strats

        Psi_accum = np.zeros((L, L))
        P_accum = np.zeros((L, L))

        this_mc_fc_all = np.zeros(steps)
        this_mc_fc_eff = np.zeros(steps)

        for t in range(steps):

            p_round, psi_round = calculate_payoffs_v2(
                S, P_accum, params['r'], params['c'],
                params['alpha'], params['theta'], params['mu']
            )

            Psi_accum += psi_round
            # Settle resources before assessing freezing and updating strategies.
            P_accum += p_round

            p_prev = np.copy(p_round)

            S, c2d, d2c, is_frozen = update_strategies_fermi_v2(S, p_round, P_accum, params['K_fermi'])

            # Measure updated strategies with the freezing mask used in this round.
            n_occ = np.sum(S < 2)
            if n_occ > 0:
                n_c = np.sum(S == 0)
                n_f = np.sum(is_frozen)
                n_act = n_occ - n_f

                n_c_act = np.sum((S == 0) & (~is_frozen))

                fc_all = n_c / n_occ
                fc_eff = n_c_act / n_occ
                this_mc_fc_all[t] = fc_all
                this_mc_fc_eff[t] = fc_eff

                fc_all_avg[t] += fc_all

                fc_eff_avg[t] += n_c_act / n_occ

                phi_f_avg[t] += n_f / n_occ

                # Fully frozen realizations contribute zero to active-cooperation averages.
                if n_act > 0:
                    fc_act_avg[t] += n_c_act / n_act
                c2d_avg[t] += c2d / n_occ
                d2c_avg[t] += d2c / n_occ

        ss_fc_all_list.append(np.mean(this_mc_fc_all[-100:]))
        ss_fc_eff_list.append(np.mean(this_mc_fc_eff[-100:]))

    avg_fc_all = np.mean(ss_fc_all_list)
    max_fc_all = np.max(ss_fc_all_list)
    min_fc_all = np.min(ss_fc_all_list)
    std_fc_all = np.std(ss_fc_all_list)

    avg_fc_eff = np.mean(ss_fc_eff_list)

    print(f"📊 [Summary Log] r={curr_r:.2f}, μ={curr_mu:.2f}, α={curr_alpha:.2f}, θ={curr_theta:.2f}")
    print(f"    - Avg Overall fc: {avg_fc_all:.4f} | Avg Effective fc: {avg_fc_eff:.4f}")
    print(f"    - Range: [{min_fc_all:.4f}, {max_fc_all:.4f}]")
    print("-" * 60)

    return fc_all_avg / mc, fc_eff_avg / mc, phi_f_avg / mc, fc_act_avg / mc, rho_avg / mc, c2d_avg / mc, d2c_avg / mc
