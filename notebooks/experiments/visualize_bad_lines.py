### IMPORTS ###

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

###########################

plt.rcParams.update({
    'axes.titlesize': 23,
    'axes.labelsize': 20,
    'legend.fontsize': 17,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.titlesize': 30,
    'axes.titlepad': 18,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
BAD_LINES_BY_LLM_PATH = PROJECT_ROOT / "analysis" / "bad_lines" / "bad_lines_by_llm.csv"


FONT_CONFIG = {
    'suptitle_size': 30, 
    'suptitle_weight': 'bold',
    'title_size': 23,  
    'title_weight': 'normal',
    'label_size': 20,  
    'tick_size': 20,  
    'bar_label_size': 15,
    'legend_size': 20, 
    'legend_title_size': 20,
    'text_color_dark': '#333333',
    'text_color_medium': '#666666',
    'text_color_light': '#CCCCCC',
}

"""
Plot bad lines by error type, shot, and domain for each LLM.
"""

def plot_faceted_by_llm(df):
    
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
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows), sharex=True, sharey=False)
        axes = axes.flatten()
        
        max_value = subset['bad_rows_count'].max() if len(subset) > 0 else 1
        y_max = max_value * 1.1

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
                width=1.0
            )
            
            ax.set_ylim(bottom=0, top=y_max)

            ax.set_title(domain.title(), fontsize=FONT_CONFIG['title_size'], fontweight=FONT_CONFIG['title_weight'], color=FONT_CONFIG['text_color_dark'], pad=10)
            
            if i >= (n_rows - 1) * n_cols:
                ax.set_xlabel('Shot Type', fontsize=FONT_CONFIG['label_size'], color=FONT_CONFIG['text_color_medium'])
            else:
                ax.set_xlabel('')
            

            if i % n_cols == 0:
                ax.set_ylabel('Number of Bad Rows', fontsize=FONT_CONFIG['label_size'], color=FONT_CONFIG['text_color_medium'])
            else:
                ax.set_ylabel('')

            if i == 0:
                ax.tick_params(axis='y', colors=FONT_CONFIG['text_color_medium'], labelsize=FONT_CONFIG['tick_size'], left=True, labelleft=True)
            else:
                ax.tick_params(axis='y', colors=FONT_CONFIG['text_color_medium'], labelsize=FONT_CONFIG['tick_size'], left=True, labelleft=False)

            ax.set_xticks(range(len(shot_order)))
            if i >= (n_rows - 1) * n_cols:
                ax.set_xticklabels([s.title() for s in shot_order], fontsize=FONT_CONFIG['tick_size'], color=FONT_CONFIG['text_color_medium'])
                ax.tick_params(colors=FONT_CONFIG['text_color_medium'], width=0.5, length=3, bottom=True, labelbottom=True)
            else:
                ax.set_xticklabels([])
                ax.tick_params(colors=FONT_CONFIG['text_color_medium'], width=0.5, length=3, bottom=True, labelbottom=False)

            ax.grid(axis='y', linestyle='-', alpha=0.15, linewidth=0.5, color=FONT_CONFIG['text_color_light'], zorder=1)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color(FONT_CONFIG['text_color_light'])
                ax.spines[spine].set_linewidth(0.5)

            for container in ax.containers:
                labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
                ax.bar_label(container, labels=labels, padding=2, fontsize=FONT_CONFIG['bar_label_size'], color=FONT_CONFIG['text_color_dark'], zorder=3)

            if ax.get_legend():
                ax.get_legend().remove()

        for j in range(n_domains, len(axes)):
            axes[j].set_visible(False)

        all_handles = []
        all_labels = []
        seen_labels = set()
        
        visible_axes = [ax for ax in axes[:n_domains]]
        
        for ax in axes:
            if ax.containers:
                try:
                    handles, labels = ax.get_legend_handles_labels()
                    for h, lbl in zip(handles, labels):
                        if lbl not in seen_labels:
                            all_handles.append(h)
                            all_labels.append(lbl)
                            seen_labels.add(lbl)
                except:
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
                    bbox_to_anchor=(center_x, -0.15),
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
                    bbox_to_anchor=(0.5, -0.15),
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
            fontsize=FONT_CONFIG['suptitle_size'], fontweight=FONT_CONFIG['suptitle_weight'], x=title_x, y=0.995, color=FONT_CONFIG['text_color_dark']
        )

        plt.tight_layout(rect=[0, 0.12, 1, 0.97])
        
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

        print("\n" + "=" * 80)
        print("AVERAGE BAD ROWS PER LLM (MEAN PER RUN ACROSS ALL DOMAINS)")
        print("=" * 80)

        llm_run_totals = (
            stats_df.groupby(['llm', 'run'])['bad_rows_count']
            .sum()
            .unstack('run')
            .reindex(columns=run_levels, fill_value=0)
            .fillna(0)
        )

        llm_run_stats = llm_run_totals.copy()
        llm_run_stats['avg_bad_rows_per_run'] = llm_run_totals.mean(axis=1)
        llm_run_stats['std_bad_rows_per_run'] = llm_run_totals.std(axis=1, ddof=1)
        llm_run_stats['min_bad_rows_per_run'] = llm_run_totals.min(axis=1)
        llm_run_stats['max_bad_rows_per_run'] = llm_run_totals.max(axis=1)
        llm_run_stats['num_runs'] = llm_run_totals.shape[1]

        llm_run_stats = (
            llm_run_stats[['avg_bad_rows_per_run', 'std_bad_rows_per_run',
                           'min_bad_rows_per_run', 'max_bad_rows_per_run', 'num_runs']]
            .reset_index()
            .sort_values('avg_bad_rows_per_run', ascending=False)
        )

        print(
            llm_run_stats.to_string(
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

        if has_domain_info:
            print("\n" + "=" * 80)
            print("AVERAGE BAD ROWS PER LLM BY DOMAIN")
            print("=" * 80)

            domain_run_totals = (
                stats_df.groupby(['llm', 'domain', 'run'])['bad_rows_count']
                .sum()
                .unstack('run')
                .reindex(columns=run_levels, fill_value=0)
                .fillna(0)
            )

            domain_avg = (
                domain_run_totals.mean(axis=1)
                .reset_index(name='avg_bad_rows_per_run')
            )

            for llm in llm_run_stats['llm']:
                subset = (
                    domain_avg[domain_avg['llm'] == llm]
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
            print("AVERAGE BAD ROWS PER LLM BY SHOT")
            print("=" * 80)

            shot_run_totals = (
                stats_df.groupby(['llm', 'shot', 'run'])['bad_rows_count']
                .sum()
                .unstack('run')
                .reindex(columns=run_levels, fill_value=0)
                .fillna(0)
            )

            shot_avg = (
                shot_run_totals.mean(axis=1)
                .reset_index(name='avg_bad_rows_per_run')
            )

            for llm in llm_run_stats['llm']:
                subset = (
                    shot_avg[shot_avg['llm'] == llm]
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

        if has_domain_shot_info:
            print("\n" + "=" * 80)
            print("AVERAGE BAD ROWS PER LLM BY DOMAIN AND SHOT")
            print("=" * 80)

            domain_shot_run_totals = (
                stats_df.groupby(['llm', 'domain', 'shot', 'run'])['bad_rows_count']
                .sum()
                .unstack('run')
                .reindex(columns=run_levels, fill_value=0)
                .fillna(0)
            )

            domain_shot_avg = (
                domain_shot_run_totals.mean(axis=1)
                .reset_index(name='avg_bad_rows_per_run')
            )

            if 'shot' in domain_shot_avg.columns:
                domain_shot_avg['shot'] = pd.Categorical(
                    domain_shot_avg['shot'],
                    categories=shot_order,
                    ordered=True
                )

            for llm in llm_run_stats['llm']:
                subset = (
                    domain_shot_avg[domain_shot_avg['llm'] == llm]
                    .sort_values(['domain', 'shot'])
                )
                if subset.empty:
                    continue
                print(f"\n{llm}:")
                print(
                    subset[['domain', 'shot', 'avg_bad_rows_per_run']].to_string(
                        index=False,
                        formatters={'avg_bad_rows_per_run': '{:,.2f}'.format}
                    )
                )

    print("=" * 80)


def load_bad_lines_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load the bad-lines CSV.
    """
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