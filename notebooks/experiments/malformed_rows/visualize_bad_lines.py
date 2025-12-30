''' Visualize malformed rows found in LLM-generated datasets. '''

### IMPORTS ###

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

###########################

sns.set_style("whitegrid")

plt.rcParams.update({
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
})


def find_project_root():
    """Find the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "README.md").exists():
            return current
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
BAD_LINES_BY_LLM_PATH = PROJECT_ROOT / "analysis" / "bad_lines" / "bad_lines_by_llm.csv"


FONT_CONFIG = {
    'suptitle_size': 36,
    'title_size': 33,
    'title_weight': 'normal',
    'label_size': 24,
    'xlabel_size': 29,
    'ylabel_size': 29,
    'tick_size': 24,
    'xtick_label_size': 30,
    'ytick_label_size': 30,
    'bar_label_size': 15,
    'legend_size': 24,
    'legend_title_size': 24,
    'text_color_dark': '#080808',
    'text_color_medium': '#080808',
    'text_color_light': '#808080',
}


def format_domain_name(domain: str) -> str:
    if domain.lower() == 'hatecrime':
        return 'Hate Crime'
    return domain.title()


def format_error_type(error_type: str) -> str:
    '''Format error type labels by removing underscores and applying title case.'''
    if not error_type or error_type == 'Other':
        return error_type
    return error_type.replace('_', ' ').title()


def plot_faceted_by_llm(df):
    '''Plot bad rows by error type, shot, and domain for each LLM.'''

    agg_data = df.groupby(['llm', 'domain', 'shot', 'error_type'])['bad_rows_count'].sum().reset_index()

    shot_order = ['zero', 'one', 'few']
    error_types = sorted(agg_data['error_type'].unique())

    base_palette = sns.color_palette("Set2", n_colors=len(error_types))
    palette = [(r * 0.9, g * 0.9, b * 0.9) for r, g, b in base_palette]

    llms = agg_data['llm'].unique()

    domain_order = ["hatecrime", "employment", "lending"]

    for llm in llms:
        subset = agg_data[agg_data['llm'] == llm]
        domains = [d for d in domain_order if d in subset['domain'].unique()]
        remaining = [d for d in subset['domain'].unique() if d not in domains]
        domains.extend(sorted(remaining))

        n_domains = len(domains)
        n_cols = 3
        n_rows = (n_domains + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(18, 6 * n_rows),
            sharex=True,
            sharey=False
        )
        axes = axes.flatten()

        y_max = 2000
        y_ticks = [0, 500, 1000, 1500, 2000]

        for i, domain in enumerate(domains):
            ax = axes[i]
            dom_data = subset[subset['domain'] == domain]

            sns.barplot(
                data=dom_data,
                x='shot',
                y='bad_rows_count',
                hue='error_type',
                hue_order=error_types,
                palette=palette,
                errorbar=None,
                edgecolor='white',
                linewidth=0.5,
                ax=ax,
                saturation=0.85,
                zorder=2,
                width=0.7
            )

            ax.set_ylim(bottom=0, top=y_max)
            ax.set_yticks(y_ticks)

            ax.set_title(
                format_domain_name(domain),
                fontsize=FONT_CONFIG['title_size'],
                fontweight=FONT_CONFIG['title_weight'],
                color='black',
                pad=25
            )

            if i % n_cols == 0:
                ax.set_ylabel(
                    'Number of Bad Rows',
                    fontsize=FONT_CONFIG['ylabel_size'],
                    color='black'
                )
            else:
                ax.set_ylabel('')

            if i == 0:
                ax.tick_params(
                    axis='y',
                    colors='black',
                    labelsize=FONT_CONFIG['ytick_label_size'],
                    left=True,
                    labelleft=True
                )
            else:
                ax.tick_params(
                    axis='y',
                    colors='black',
                    labelsize=FONT_CONFIG['ytick_label_size'],
                    left=True,
                    labelleft=False
                )

            ax.set_xticks(range(len(shot_order)))
            ax.set_xticklabels(
                [s.lower() for s in shot_order],
                fontsize=FONT_CONFIG['xtick_label_size'],
                color='black'
            )
            ax.tick_params(
                colors='black',
                width=0.5,
                length=3,
                bottom=True,
                labelbottom=True
            )
            ax.set_xlim(-0.5, len(shot_order) - 0.5)

            ax.grid(
                True,
                axis='y',
                alpha=1.0,
                linestyle='-',
                linewidth=0.7,
                zorder=0
            )
            ax.set_axisbelow(True)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color(FONT_CONFIG['text_color_light'])
                ax.spines[spine].set_linewidth(0.5)

            if ax.get_legend():
                ax.get_legend().remove()

        for j in range(n_domains, len(axes)):
            axes[j].set_visible(False)

        for ax in axes:
            ax.set_xlabel('')

        visible_axes_for_x = [ax for ax in axes if ax.get_visible()]
        if visible_axes_for_x:
            bboxes_x = [ax.get_position() for ax in visible_axes_for_x]
            leftmost_x = min(bb.x0 for bb in bboxes_x)
            rightmost_x = max(bb.x1 for bb in bboxes_x)
            center_x_label = (leftmost_x + rightmost_x) / 2
        else:
            center_x_label = 0.5

        all_handles = []
        all_labels = []
        seen_labels = set()

        visible_axes = [ax for ax in axes[:n_domains]]

        for ax in axes:
            if ax.containers:
                try:
                    handles, labels = ax.get_legend_handles_labels()
                    for h, lbl in zip(handles, labels):
                        formatted_lbl = format_error_type(lbl)
                        if formatted_lbl not in seen_labels and formatted_lbl != '':
                            all_handles.append(h)
                            all_labels.append(formatted_lbl)
                            seen_labels.add(formatted_lbl)
                except Exception:
                    pass

        if all_handles and all_labels:
            if visible_axes:
                bboxes = [ax.get_position() for ax in visible_axes]
                leftmost = min(bb.x0 for bb in bboxes)
                rightmost = max(bb.x1 for bb in bboxes)
                center_x = (leftmost + rightmost) / 2

                fig.legend(
                    all_handles, all_labels,
                    title="Error Type",
                    loc='lower center',
                    bbox_to_anchor=(center_x, -0.30),
                    ncol=min(len(error_types), 6),
                    fontsize=FONT_CONFIG['legend_size'],
                    title_fontsize=FONT_CONFIG['legend_title_size'],
                    frameon=True,
                    framealpha=0.9
                )
            else:
                fig.legend(
                    all_handles, all_labels,
                    title="Error Type",
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.30),
                    ncol=min(len(error_types), 6),
                    fontsize=FONT_CONFIG['legend_size'],
                    title_fontsize=FONT_CONFIG['legend_title_size'],
                    frameon=True,
                    framealpha=0.9
                )

        if visible_axes:
            bboxes = [ax.get_position() for ax in visible_axes]
            leftmost = min(bb.x0 for bb in bboxes)
            rightmost = max(bb.x1 for bb in bboxes)
            title_x = (leftmost + rightmost) / 2
        else:
            title_x = 0.5

        fig.suptitle(
            f"{llm}",
            fontsize=FONT_CONFIG['suptitle_size'],
            x=title_x,
            y=0.995,
            color='black'
        )

        plt.tight_layout(rect=[0, 0.12, 1, 0.97])

        fig.text(
            center_x_label,
            0.04,
            "Shot Type",
            ha="center",
            va="center",
            fontsize=FONT_CONFIG['xlabel_size'],
            color="black"
        )

        llm_name_clean = llm.replace('/', '_').replace('-', '_').replace('.', '_')
        filename = f"bad_rows_by_error_type_shot_domain_{llm_name_clean}.png"
        filepath = FIGURES_DIR / filename
        fig.savefig(filepath, bbox_inches='tight', dpi=300)

        plt.show()

    print("\n" + "=" * 80)
    print("SUMMARY: Total Bad Lines Found Per LLM")
    print("=" * 80)

    llm_summary = agg_data.groupby('llm')['bad_rows_count'].sum().reset_index()
    llm_summary.columns = ['llm', 'total_bad_rows']

    if BAD_LINES_BY_LLM_PATH.exists():
        llm_total_rows_df = pd.read_csv(
            BAD_LINES_BY_LLM_PATH,
            usecols=['llm', 'total_bad_rows_llm', 'total_rows_llm', 'bad_rows_pct_llm'],
        ).drop_duplicates(subset='llm')
        llm_summary = llm_summary.merge(llm_total_rows_df, on='llm', how='left')
    else:
        llm_summary['total_bad_rows_llm'] = np.nan
        llm_summary['total_rows_llm'] = np.nan
        llm_summary['bad_rows_pct_llm'] = np.nan

    total_all_bad = llm_summary['total_bad_rows'].sum()

    for _, row in llm_summary.iterrows():
        llm_name = row['llm']
        total_bad = row['total_bad_rows']
        pct = (total_bad / total_all_bad * 100) if total_all_bad > 0 else 0
        total_rows_llm = row.get('total_rows_llm', np.nan)
        pct_of_llm_total = (
            row['bad_rows_pct_llm']
            if not pd.isna(row.get('bad_rows_pct_llm', np.nan))
            else (
                (total_bad / total_rows_llm * 100)
                if not pd.isna(total_rows_llm) and total_rows_llm > 0
                else np.nan
            )
        )

        if (
            not pd.isna(total_rows_llm)
            and total_rows_llm > 0
            and not pd.isna(pct_of_llm_total)
        ):
            total_bad_known = row.get('total_bad_rows_llm', total_bad)
            total_bad_display = (
                int(total_bad_known)
                if not pd.isna(total_bad_known)
                else int(total_bad)
            )
            print(
                f"{llm_name}: {total_bad_display:,} bad rows "
                f"({pct:.1f}% of all bad lines, "
                f"{pct_of_llm_total:.2f}% of {int(total_rows_llm):,} generated rows)"
            )
        else:
            print(f"{llm_name}: {int(total_bad):,} bad rows ({pct:.1f}% of all bad lines)")

    print(f"\nTotal bad lines across all LLMs: {int(total_all_bad)}")

    stats_df = df
    has_run_info = {'llm', 'run'}.issubset(stats_df.columns)
    has_domain_info = {'llm', 'domain', 'run'}.issubset(stats_df.columns)
    has_domain_shot_info = {'llm', 'domain', 'shot', 'run'}.issubset(stats_df.columns)

    if not has_run_info:
        try:
            reference_df = load_bad_lines_dataframe()
        except FileNotFoundError:
            reference_df = None

        if reference_df is not None and {'llm', 'run'}.issubset(reference_df.columns):
            if 'llm' in df.columns:
                reference_df = reference_df[reference_df['llm'].isin(df['llm'].unique())]
            if 'domain' in df.columns and 'domain' in reference_df.columns:
                reference_df = reference_df[reference_df['domain'].isin(df['domain'].unique())]
            if 'shot' in df.columns and 'shot' in reference_df.columns:
                reference_df = reference_df[reference_df['shot'].isin(df['shot'].unique())]

            stats_df = reference_df
            has_run_info = True
            has_domain_info = {'llm', 'domain', 'run'}.issubset(stats_df.columns)
            has_domain_shot_info = {'llm', 'domain', 'shot', 'run'}.issubset(stats_df.columns)

    def _sorted_runs(run_values: pd.Index) -> list:
        def sort_key(label):
            if isinstance(label, str) and label.lower().startswith("run"):
                suffix = label[3:]
                return int(suffix) if suffix.isdigit() else label
            return label
        return sorted(run_values, key=sort_key)

    if has_run_info:
        run_levels = _sorted_runs(pd.Index(stats_df['run'].unique()))

        if 'domain' not in stats_df.columns:
            stats_df = stats_df.copy()
            stats_df['domain'] = 'all_domains'
        if 'shot' not in stats_df.columns:
            stats_df = stats_df.copy()
            stats_df['shot'] = 'all_shots'

        domain_shot_run_totals = (
            stats_df.groupby(['llm', 'domain', 'shot', 'run'], as_index=False)['bad_rows_count']
            .sum()
        )

        llm_levels = pd.Index(stats_df['llm'].unique())
        domain_levels = pd.Index(stats_df['domain'].unique())
        shot_levels = pd.Index(stats_df['shot'].unique())
        run_levels_idx = pd.Index(run_levels)

        full_index = pd.MultiIndex.from_product(
            [llm_levels, domain_levels, shot_levels, run_levels_idx],
            names=['llm', 'domain', 'shot', 'run']
        )

        domain_shot_run_totals = (
            domain_shot_run_totals
            .set_index(['llm', 'domain', 'shot', 'run'])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        domain_shot_avg = (
            domain_shot_run_totals.groupby(['llm', 'domain', 'shot'], as_index=False)['bad_rows_count']
            .mean()
            .rename(columns={'bad_rows_count': 'avg_bad_rows_per_run'})
        )

        domain_shot_stats = (
            domain_shot_run_totals.groupby(['llm', 'domain', 'shot'])['bad_rows_count']
            .agg(
                avg_bad_rows_per_run='mean',
                std_bad_rows_per_run=lambda x: x.std(ddof=1),
                min_bad_rows_per_run='min',
                max_bad_rows_per_run='max',
                num_runs='size'  
            )
            .reset_index()
        )

        llm_overall = (
            domain_shot_avg.groupby('llm', as_index=False)['avg_bad_rows_per_run']
            .mean()
            .rename(columns={'avg_bad_rows_per_run': 'avg_bad_rows_per_run_over_domain_shot'})
            .sort_values('avg_bad_rows_per_run_over_domain_shot', ascending=False)
        )

        print("\n" + "=" * 80)
        print("AVERAGE BAD ROWS PER LLM (MEAN OVER DOMAIN-SHOT OF MEAN PER RUN)")
        print("=" * 80)
        print(
            llm_overall.to_string(
                index=False,
                formatters={'avg_bad_rows_per_run_over_domain_shot': '{:,.2f}'.format}
            )
        )

        if has_domain_info or 'domain' in stats_df.columns:
            print("\n" + "=" * 80)
            print("AVERAGE BAD ROWS PER LLM BY DOMAIN (MEAN OVER SHOT OF MEAN PER RUN)")
            print("=" * 80)

            llm_domain_avg = (
                domain_shot_avg.groupby(['llm', 'domain'], as_index=False)['avg_bad_rows_per_run']
                .mean()
                .rename(columns={'avg_bad_rows_per_run': 'avg_bad_rows_per_run'})
            )

            for llm in llm_overall['llm']:
                subset = (
                    llm_domain_avg[llm_domain_avg['llm'] == llm]
                    .sort_values('domain')
                )
                if subset.empty:
                    continue
                print(f"\n{llm}:")
                print(
                    subset[['domain', 'avg_bad_rows_per_run']].to_string(
                        index=False,
                        formatters={'avg_bad_rows_per_run': '{:,.2f}'.format}
                    )
                )
                
        if 'shot' in stats_df.columns:
            print("\n" + "=" * 80)
            print("AVERAGE BAD ROWS PER LLM BY SHOT (MEAN OVER DOMAIN OF MEAN PER RUN)")
            print("=" * 80)

            llm_shot_avg = (
                domain_shot_avg.groupby(['llm', 'shot'], as_index=False)['avg_bad_rows_per_run']
                .mean()
                .rename(columns={'avg_bad_rows_per_run': 'avg_bad_rows_per_run'})
            )

            for llm in llm_overall['llm']:
                subset = (
                    llm_shot_avg[llm_shot_avg['llm'] == llm]
                    .sort_values('shot')
                )
                if subset.empty:
                    continue
                print(f"\n{llm}:")
                print(
                    subset[['shot', 'avg_bad_rows_per_run']].to_string(
                        index=False,
                        formatters={'avg_bad_rows_per_run': '{:,.2f}'.format}
                    )
                )


        if has_domain_shot_info or ('domain' in stats_df.columns and 'shot' in stats_df.columns):
            print("\n" + "=" * 80)
            print("AVERAGE BAD ROWS PER LLM BY DOMAIN AND SHOT (MEAN PER RUN)")
            print("=" * 80)

            domain_shot_stats_print = domain_shot_stats.copy()
            domain_shot_stats_print['shot'] = pd.Categorical(
                domain_shot_stats_print['shot'],
                categories=shot_order,
                ordered=True
            )

            for llm in llm_overall['llm']:
                subset = (
                    domain_shot_stats_print[domain_shot_stats_print['llm'] == llm]
                    .sort_values(['domain', 'shot'])
                )
                if subset.empty:
                    continue
                print(f"\n{llm}:")
                print(
                    subset[['domain', 'shot', 'avg_bad_rows_per_run', 'std_bad_rows_per_run',
                            'min_bad_rows_per_run', 'max_bad_rows_per_run', 'num_runs']]
                    .to_string(
                        index=False,
                        formatters={
                            'avg_bad_rows_per_run': '{:,.2f}'.format,
                            'std_bad_rows_per_run': (lambda x: '{:,.2f}'.format(x) if not pd.isna(x) else 'nan'),
                            'min_bad_rows_per_run': '{:,.0f}'.format,
                            'max_bad_rows_per_run': '{:,.0f}'.format,
                            'num_runs': '{:d}'.format
                        }
                    )
                )

    print("=" * 80)


def load_bad_lines_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    '''Load the bad-lines CSV.'''
    default_path = (
        PROJECT_ROOT
        / "analysis"
        / "bad_lines"
        / "bad_lines_by_llm_domain_run_shot.csv"
    )

    path = Path(csv_path) if csv_path else default_path
    if not path.exists():
        raise FileNotFoundError(
            f"No bad-lines CSV found at {path}. "
            "Provide --csv pointing to one of the aggregated files under analysis/bad_lines."
        )

    return pd.read_csv(path)
