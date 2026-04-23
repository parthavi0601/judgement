"""
Statistical analysis of Hybrid vs Pure NFSP game logs.
Focus: WIN RATE comparison only (no payoff analysis).

Primary test: McNemar's test on paired win/loss outcomes.
Secondary: Clopper-Pearson CIs, bootstrap CIs on win-rate diff, per-seat breakdown.

Usage:
    python analyze_results.py --log-dir ./logs
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.1)
HYBRID_COLOR = '#2196F3'
PURE_COLOR   = '#FF5722'
ACCENT       = '#4CAF50'



def winrate_ci(wins, total, ci=0.95):
    """Clopper-Pearson exact CI for a proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    alpha = 1 - ci
    lo = stats.beta.ppf(alpha / 2, wins, total - wins + 1) if wins > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, wins + 1, total - wins) if wins < total else 1.0
    return wins / total, lo, hi



def bootstrap_winrate_diff_ci(h_wins_arr, p_wins_arr, n_boot=10000, ci=0.95, seed=42):
    """
    Bootstrap CI on the difference in win rates (Hybrid - Pure).
    h_wins_arr, p_wins_arr: arrays of 0/1 (per-deal win indicators).
    """
    rng = np.random.RandomState(seed)
    n = len(h_wins_arr)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        h_wr = h_wins_arr[idx].mean()
        p_wr = p_wins_arr[idx].mean()
        diffs.append(h_wr - p_wr)
    diffs = np.array(diffs)
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return lo, hi, diffs



def mcnemar_test(h_wins, p_wins):
    """
    McNemar's test for paired binary outcomes.
    Compares whether Hybrid wins more often than Pure on the SAME deals.

    Returns dict with test statistic, p-value, and contingency counts.
    """
    both_win  = int(((h_wins == 1) & (p_wins == 1)).sum())
    h_only    = int(((h_wins == 1) & (p_wins == 0)).sum())  # discordant: H wins, P loses
    p_only    = int(((h_wins == 0) & (p_wins == 1)).sum())  # discordant: H loses, P wins
    both_lose = int(((h_wins == 0) & (p_wins == 0)).sum())

    n_disc = h_only + p_only
    if n_disc == 0:
        return {
            'both_win': both_win, 'h_only': h_only, 'p_only': p_only, 'both_lose': both_lose,
            'n_discordant': 0, 'chi2': 0.0, 'p_value': 1.0,
        }

    chi2 = (abs(h_only - p_only) - 1) ** 2 / n_disc if n_disc > 0 else 0.0
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        'both_win': both_win,
        'h_only': h_only,
        'p_only': p_only,
        'both_lose': both_lose,
        'n_discordant': n_disc,
        'chi2': chi2,
        'p_value': p_value,
    }



def load_hvp_data(log_dir):
    """Load hybrid_vs_pure_log.csv and pivot into paired arrays."""
    path = os.path.join(log_dir, 'hybrid_vs_pure_log.csv')
    if not os.path.exists(path):
        print(f'  [SKIP] {path} not found — no Hybrid vs Pure data.')
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f'  [SKIP] {path} is empty.')
        return None
    return df


def load_eval_data(log_dir):
    """Load eval_games_log.csv."""
    path = os.path.join(log_dir, 'eval_games_log.csv')
    if not os.path.exists(path):
        print(f'  [SKIP] {path} not found — no eval data.')
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f'  [SKIP] {path} is empty.')
        return None
    return df


# ======================================================================
#  HYBRID vs PURE ANALYSIS  (win-rate focused)
# ======================================================================

def analyze_hvp(df, plot_dir, report_lines):
    """Full paired win-rate analysis of Hybrid vs Pure games."""
    report_lines.append('=' * 72)
    report_lines.append('  HYBRID vs PURE — WIN RATE STATISTICAL ANALYSIS')
    report_lines.append('=' * 72)

    # ── Build paired win arrays ──
    hybrid_rows = df[df['agent_type'] == 'hybrid']
    pure_rows   = df[df['agent_type'] == 'pure']

    # Focus player = the one in the hybrid seat
    h_focus = hybrid_rows[hybrid_rows['player_id'] == hybrid_rows['seat']]
    p_focus = pure_rows[pure_rows['player_id'] == pure_rows['seat']]

    h_focus = h_focus.sort_values(['seat', 'game_id']).reset_index(drop=True)
    p_focus = p_focus.sort_values(['seat', 'game_id']).reset_index(drop=True)

    n = min(len(h_focus), len(p_focus))
    if n == 0:
        report_lines.append('  No paired data found.')
        return
    h_focus = h_focus.iloc[:n]
    p_focus = p_focus.iloc[:n]

    h_wins = h_focus['win'].values
    p_wins = p_focus['win'].values

    # ── 1. OVERALL WIN RATES with CIs ──
    h_total_wins = int(h_wins.sum())
    p_total_wins = int(p_wins.sum())
    h_wr, h_lo, h_hi = winrate_ci(h_total_wins, n)
    p_wr, p_lo, p_hi = winrate_ci(p_total_wins, n)
    wr_diff = (h_wr - p_wr) * 100

    report_lines.append('')
    report_lines.append(f'  Total paired deals: {n}')
    report_lines.append('')
    report_lines.append('  ┌───────────────────────────────────────────────────────┐')
    report_lines.append('  │              WIN RATES  (Clopper-Pearson 95% CI)      │')
    report_lines.append('  ├───────────────────────────────────────────────────────┤')
    report_lines.append(f'  │  Hybrid MC-NFSP : {h_wr*100:5.1f}%  [{h_lo*100:5.1f}%, {h_hi*100:5.1f}%]           │')
    report_lines.append(f'  │  Pure NFSP      : {p_wr*100:5.1f}%  [{p_lo*100:5.1f}%, {p_hi*100:5.1f}%]           │')
    report_lines.append(f'  │  Difference     : {wr_diff:+5.1f} percentage points              │')
    report_lines.append('  └───────────────────────────────────────────────────────┘')
    report_lines.append('')

    # ── 2. McNEMAR'S TEST (paired binary) ──
    mc = mcnemar_test(h_wins, p_wins)
    report_lines.append('  ┌───────────────────────────────────────────────────────┐')
    report_lines.append('  │         McNEMAR\'S TEST  (paired win/loss)             │')
    report_lines.append('  ├───────────────────────────────────────────────────────┤')
    report_lines.append(f'  │  Both win  (concordant)  : {mc["both_win"]:>6d}                     │')
    report_lines.append(f'  │  Hybrid wins, Pure loses : {mc["h_only"]:>6d}  (discordant)        │')
    report_lines.append(f'  │  Pure wins, Hybrid loses : {mc["p_only"]:>6d}  (discordant)        │')
    report_lines.append(f'  │  Both lose (concordant)  : {mc["both_lose"]:>6d}                     │')
    report_lines.append(f'  │  χ² statistic            : {mc["chi2"]:>10.4f}                 │')
    report_lines.append(f'  │  p-value                 : {mc["p_value"]:>10.6f}                 │')
    report_lines.append('  └───────────────────────────────────────────────────────┘')
    sig = '***' if mc['p_value'] < 0.001 else '**' if mc['p_value'] < 0.01 else '*' if mc['p_value'] < 0.05 else 'n.s.'
    report_lines.append(f'  Significance: {sig}  (*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant)')
    report_lines.append('')

    # ── 3. BOOTSTRAP CI on win-rate difference ──
    b_lo, b_hi, boot_diffs = bootstrap_winrate_diff_ci(h_wins, p_wins, n_boot=10000)
    report_lines.append(f'  Bootstrap 95% CI on win-rate difference (H − P), 10,000 resamples:')
    report_lines.append(f'    [{b_lo*100:+.2f}%, {b_hi*100:+.2f}%]')
    report_lines.append('')

    # ── 4. PER-SEAT BREAKDOWN ──
    report_lines.append(f'  Per-Seat Win Rate:')
    report_lines.append(f'  {"Seat":<6} {"N":>5} {"Hybrid WR":>12} {"Pure WR":>12} {"Δ WR":>10}')
    report_lines.append(f'  {"─"*6} {"─"*5} {"─"*12} {"─"*12} {"─"*10}')
    for seat in sorted(df['seat'].unique()):
        h_s = h_focus[h_focus['seat'] == seat]['win'].values
        p_s = p_focus[p_focus['seat'] == seat]['win'].values
        ns = min(len(h_s), len(p_s))
        if ns == 0:
            continue
        h_s, p_s = h_s[:ns], p_s[:ns]
        h_seat_wr = h_s.mean() * 100
        p_seat_wr = p_s.mean() * 100
        delta = h_seat_wr - p_seat_wr
        report_lines.append(
            f'  {seat:<6} {ns:>5} {h_seat_wr:>11.1f}% {p_seat_wr:>11.1f}% {delta:>+9.1f}%'
        )
    report_lines.append('')

    # ── PLOTS ──
    os.makedirs(plot_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    #  Plot 1: Win Rate Bar Chart with 95% CIs
    #  "Which agent wins more often? Bars show overall win rate,
    #   error bars show 95% confidence interval."
    # ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Hybrid MC-NFSP', 'Pure NFSP']
    means = [h_wr * 100, p_wr * 100]
    ci_lo_vals = [means[0] - h_lo * 100, means[1] - p_lo * 100]
    ci_hi_vals = [h_hi * 100 - means[0], p_hi * 100 - means[1]]
    bars = ax.bar(labels, means, color=[HYBRID_COLOR, PURE_COLOR], alpha=0.85,
                  edgecolor='white', linewidth=1.5, width=0.5)
    ax.errorbar(labels, means, yerr=[ci_lo_vals, ci_hi_vals],
                fmt='none', ecolor='black', capsize=8, capthick=2, lw=2)
    ax.set_ylabel('Win Rate (%)', fontsize=13)
    ax.set_title(
        f'Win Rate: Hybrid MC-NFSP vs Pure NFSP\n'
        f'({n} paired deals · same cards dealt to both agents)\n'
        f'Error bars = 95% Clopper-Pearson confidence interval',
        fontsize=13, fontweight='bold'
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'winrate_comparison.png'), dpi=150)
    plt.close()

    # ──────────────────────────────────────────────────────────────────
    #  Plot 2: Per-Seat Win Rate Comparison
    #  "Does one agent do better only in certain seat positions?
    #   Each seat position is tested independently."
    # ──────────────────────────────────────────────────────────────────
    seats = sorted(df['seat'].unique())
    h_wr_seat = []
    p_wr_seat = []
    h_ci_lo_seat = []
    h_ci_hi_seat = []
    p_ci_lo_seat = []
    p_ci_hi_seat = []
    seat_ns = []
    for seat in seats:
        h_s = h_focus[h_focus['seat'] == seat]['win'].values
        p_s = p_focus[p_focus['seat'] == seat]['win'].values
        ns = min(len(h_s), len(p_s))
        h_s, p_s = h_s[:ns], p_s[:ns]
        seat_ns.append(ns)
        hw, hlo, hhi = winrate_ci(int(h_s.sum()), ns)
        pw, plo, phi = winrate_ci(int(p_s.sum()), ns)
        h_wr_seat.append(hw * 100)
        p_wr_seat.append(pw * 100)
        h_ci_lo_seat.append(hw * 100 - hlo * 100)
        h_ci_hi_seat.append(hhi * 100 - hw * 100)
        p_ci_lo_seat.append(pw * 100 - plo * 100)
        p_ci_hi_seat.append(phi * 100 - pw * 100)

    x = np.arange(len(seats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    h_bars = ax.bar(x - width / 2, h_wr_seat, width, color=HYBRID_COLOR, alpha=0.85, label='Hybrid MC-NFSP')
    p_bars = ax.bar(x + width / 2, p_wr_seat, width, color=PURE_COLOR, alpha=0.85, label='Pure NFSP')
    ax.errorbar(x - width / 2, h_wr_seat, yerr=[h_ci_lo_seat, h_ci_hi_seat],
                fmt='none', ecolor='black', capsize=5, capthick=1.5)
    ax.errorbar(x + width / 2, p_wr_seat, yerr=[p_ci_lo_seat, p_ci_hi_seat],
                fmt='none', ecolor='black', capsize=5, capthick=1.5)
    ax.set_xlabel('Seat Position (player turn order)', fontsize=13)
    ax.set_ylabel('Win Rate (%)', fontsize=13)
    ax.set_title(
        f'Win Rate by Seat Position — Hybrid vs Pure\n'
        f'(Each seat tested with {seat_ns[0] if seat_ns else "?"} paired deals · 95% CIs shown)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seat {s}\n(n={seat_ns[i]})' for i, s in enumerate(seats)])
    ax.legend(fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'per_seat_winrate.png'), dpi=150)
    plt.close()

    # ──────────────────────────────────────────────────────────────────
    #  Plot 3: Bootstrap Distribution of Win-Rate Difference
    #  "If we resampled our data 10,000 times, how consistently does
    #   Hybrid beat Pure? The shaded region is the 95% CI."
    # ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(boot_diffs * 100, bins=80, color=ACCENT, alpha=0.7, edgecolor='white',
            linewidth=0.5, density=True)
    ax.axvline(0, color='gray', ls='--', lw=1.5, alpha=0.7, label='No difference (0%)')
    observed_diff = (h_wr - p_wr) * 100
    ax.axvline(observed_diff, color=HYBRID_COLOR, ls='-', lw=2,
               label=f'Observed Δ = {observed_diff:+.1f}%')
    ax.axvspan(b_lo * 100, b_hi * 100, alpha=0.15, color=HYBRID_COLOR, label='95% Bootstrap CI')
    ax.set_xlabel('Win Rate Difference: Hybrid − Pure (percentage points)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(
        f'Bootstrap Distribution of Win-Rate Difference (Hybrid − Pure)\n'
        f'10,000 resamples · 95% CI: [{b_lo*100:+.1f}%, {b_hi*100:+.1f}%]\n'
        f'If the CI excludes 0, Hybrid is significantly better.',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'bootstrap_winrate_diff.png'), dpi=150)
    plt.close()

    # ──────────────────────────────────────────────────────────────────
    #  Plot 4: McNemar Contingency Pie Chart
    #  "Out of all deals, how often did both agents agree vs disagree?"
    # ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_pie = [
        f'Both Win\n({mc["both_win"]})',
        f'Only Hybrid Wins\n({mc["h_only"]})',
        f'Only Pure Wins\n({mc["p_only"]})',
        f'Both Lose\n({mc["both_lose"]})',
    ]
    sizes = [mc['both_win'], mc['h_only'], mc['p_only'], mc['both_lose']]
    colors_pie = ['#4CAF50', HYBRID_COLOR, PURE_COLOR, '#9E9E9E']
    explode = (0, 0.05, 0.05, 0)  # highlight discordant slices
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_pie, colors=colors_pie, explode=explode,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    ax.set_title(
        f'McNemar Contingency: Same Deal, Different Outcomes\n'
        f'"Only Hybrid Wins" vs "Only Pure Wins" = discordant pairs\n'
        f'McNemar χ²={mc["chi2"]:.2f}, p={mc["p_value"]:.4f} ({sig})',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mcnemar_contingency.png'), dpi=150)
    plt.close()

    report_lines.append(f'  Plots saved to {plot_dir}/')
    report_lines.append('')

# ======================================================================
#  NFSP vs RULE-BASED ANALYSIS
# ======================================================================


# ======================================================================
#  EVAL GAMES ANALYSIS  (win-rate only)
# ======================================================================

def analyze_eval(df, plot_dir, report_lines):
    """Analyze Phase 2 (hybrid_all) vs Phase 3 (pure_all) eval games — win rate only."""
    report_lines.append('=' * 72)
    report_lines.append('  EVAL GAMES — HYBRID-ALL vs PURE-ALL (WIN RATE)')
    report_lines.append('=' * 72)

    for eval_type in ['hybrid_all', 'pure_all']:
        sub = df[df['eval_type'] == eval_type]
        if sub.empty:
            report_lines.append(f'  [{eval_type}] No data.')
            continue

        wins = int(sub['win'].sum())
        total = len(sub)
        wr, wr_lo, wr_hi = winrate_ci(wins, total)

        label = 'Hybrid (all)' if eval_type == 'hybrid_all' else 'Pure (all)'
        report_lines.append(f'')
        report_lines.append(f'  {label} — {total} player-games')
        report_lines.append(f'    Win rate : {wr*100:.1f}%  (95% CI: [{wr_lo*100:.1f}%, {wr_hi*100:.1f}%])')

    # If both types exist, do a chi-squared test on win rates
    h_eval = df[df['eval_type'] == 'hybrid_all']
    p_eval = df[df['eval_type'] == 'pure_all']
    if not h_eval.empty and not p_eval.empty:
        h_wins = int(h_eval['win'].sum())
        h_total = len(h_eval)
        p_wins = int(p_eval['win'].sum())
        p_total = len(p_eval)

        # Two-proportion z-test
        h_wr_val = h_wins / h_total
        p_wr_val = p_wins / p_total
        pooled = (h_wins + p_wins) / (h_total + p_total)
        se = np.sqrt(pooled * (1 - pooled) * (1/h_total + 1/p_total)) if pooled > 0 and pooled < 1 else 1e-9
        z = (h_wr_val - p_wr_val) / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'

        report_lines.append('')
        report_lines.append(f'  Two-proportion z-test (hybrid_all vs pure_all win rates):')
        report_lines.append(f'    z = {z:+.4f},  p = {p_val:.6f},  Significance: {sig}')

        # Plot win rate comparison
        os.makedirs(plot_dir, exist_ok=True)
        h_wr, h_lo, h_hi = winrate_ci(h_wins, h_total)
        p_wr, p_lo, p_hi = winrate_ci(p_wins, p_total)

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Hybrid-All', 'Pure-All']
        means = [h_wr * 100, p_wr * 100]
        ci_lo_vals = [means[0] - h_lo * 100, means[1] - p_lo * 100]
        ci_hi_vals = [h_hi * 100 - means[0], p_hi * 100 - means[1]]
        bars = ax.bar(labels, means, color=[HYBRID_COLOR, PURE_COLOR], alpha=0.85,
                      edgecolor='white', linewidth=1.5, width=0.5)
        ax.errorbar(labels, means, yerr=[ci_lo_vals, ci_hi_vals],
                    fmt='none', ecolor='black', capsize=8, capthick=2, lw=2)
        ax.set_ylabel('Win Rate (%)', fontsize=13)
        ax.set_title(
            f'Eval Win Rate: All-Hybrid vs All-Pure Games\n'
            f'(All 4 players use same agent type · 95% CIs shown)',
            fontsize=13, fontweight='bold'
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'eval_winrate_comparison.png'), dpi=150)
        plt.close()

    report_lines.append('')
    report_lines.append(f'  Plots saved to {plot_dir}/')
    report_lines.append('')


# ======================================================================
#  MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of game logs (win-rate focused)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory containing CSV game logs')
    args = parser.parse_args()

    log_dir  = args.log_dir
    plot_dir = os.path.join(log_dir, 'plots')
    report_lines = []

    report_lines.append('')
    report_lines.append('╔' + '═' * 70 + '╗')
    report_lines.append('║' + '  WIN RATE ANALYSIS REPORT'.center(70) + '║')
    report_lines.append('╚' + '═' * 70 + '╝')
    report_lines.append('')

    hvp_df = load_hvp_data(log_dir)
    if hvp_df is not None:
        analyze_hvp(hvp_df, plot_dir, report_lines)

    # ── Eval games ──
    eval_df = load_eval_data(log_dir)
    if eval_df is not None:
        analyze_eval(eval_df, plot_dir, report_lines)

        report_lines.append('  No data found in ' + log_dir)
        report_lines.append('  Run main.py with --hybrid-vs-pure-games N and/or --eval-games N first.')

    report_lines.append('=' * 72)
    report_lines.append('  END OF REPORT')
    report_lines.append('=' * 72)

    # Print + Save
    report = '\n'.join(report_lines)
    print(report)

    summary_path = os.path.join(log_dir, 'statistical_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(report)
    print(f'\n  Full report saved to {summary_path}')
    print(f'  Plots saved to {plot_dir}/')


if __name__ == '__main__':
    main()
