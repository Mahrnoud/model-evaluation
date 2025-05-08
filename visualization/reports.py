import os
import time
from venv import logger

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# 1. Enhanced visualizations with Seaborn
def create_enhanced_visualizations(results, output_dir, lang_prefix):
    """
    Create enhanced visualizations with better aesthetics and more insights

    Args:
        results: List of evaluation results
        output_dir: Directory to save visualizations
        lang_prefix: Language prefix for filenames
    """
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")

    # Create DataFrame for easier plotting
    df = pd.DataFrame(results)

    # 1. Distribution of metrics with KDE
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(df['bleu'], kde=True, color='blue')
    plt.axvline(df['bleu'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title('BLEU Score Distribution')
    plt.xlabel('BLEU Score')
    plt.ylabel('Count')

    plt.subplot(2, 2, 2)
    sns.histplot(df['rougeL'], kde=True, color='green')
    plt.axvline(df['rougeL'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title('ROUGE-L Score Distribution')
    plt.xlabel('ROUGE-L Score')
    plt.ylabel('Count')

    plt.subplot(2, 2, 3)
    sns.histplot(df['f1_score'], kde=True, color='purple')
    plt.axvline(df['f1_score'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title('F1 Word Match Distribution')
    plt.xlabel('F1 Score')
    plt.ylabel('Count')

    plt.subplot(2, 2, 4)
    sns.histplot(df['exact_match'], kde=True, color='orange')
    plt.axvline(df['exact_match'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title('Exact Match Score Distribution')
    plt.xlabel('Exact Match Score')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lang_prefix}_score_distributions.png'), dpi=300)
    plt.close()

    # 2. Correlation matrix of metrics with heatmap
    plt.figure(figsize=(10, 8))
    metric_cols = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'f1_score', 'exact_match']
    corr = df[metric_cols].corr()

    # Create a custom diverging colormap
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4575b4', '#white', '#d73027'])

    sns.heatmap(corr, annot=True, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Between Evaluation Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lang_prefix}_metric_correlations_heatmap.png'), dpi=300)
    plt.close()

    # 3. Quality distribution with better visualization
    quality_order = ['Excellent', 'Good', 'Partial', 'Poor']
    quality_counts = df['quality'].value_counts().reindex(quality_order)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=quality_counts.index, y=quality_counts.values,
                     palette=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])

    # Add percentage labels
    total = quality_counts.sum()
    for i, count in enumerate(quality_counts.values):
        ax.text(i, count + 0.5, f'{count / total:.1%}', ha='center')

    plt.title('Quality Distribution of Generated Answers', fontsize=14)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Quality Level', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lang_prefix}_quality_distribution.png'), dpi=300)
    plt.close()

    # 4. Performance by answer length
    plt.figure(figsize=(12, 8))

    # Calculate text lengths
    df['ref_length'] = df['reference_answer'].apply(len)
    df['gen_length'] = df['generated_answer'].apply(len)
    df['length_ratio'] = df['gen_length'] / df['ref_length']

    # Create bins for reference lengths
    bins = [0, 50, 100, 200, 300, 500, 1000, max(df['ref_length'])]
    df['length_bin'] = pd.cut(df['ref_length'], bins=bins)

    # Plot performance by length bin
    plt.subplot(2, 1, 1)
    sns.boxplot(x='length_bin', y='bleu', data=df)
    plt.title('BLEU Score by Reference Answer Length')
    plt.xlabel('Reference Answer Length (characters)')
    plt.ylabel('BLEU Score')
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    sns.boxplot(x='length_bin', y='f1_score', data=df)
    plt.title('F1 Score by Reference Answer Length')
    plt.xlabel('Reference Answer Length (characters)')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lang_prefix}_performance_by_length.png'), dpi=300)
    plt.close()

    # 5. Interactive scatter plot with plotly (if available)
    try:
        import plotly.express as px
        import plotly.io as pio

        # Create interactive scatter plot
        fig = px.scatter(df, x='bleu', y='f1_score', color='quality',
                         hover_data=['question', 'reference_answer', 'generated_answer'],
                         color_discrete_map={
                             'Excellent': '#2ecc71',
                             'Good': '#3498db',
                             'Partial': '#f39c12',
                             'Poor': '#e74c3c'
                         },
                         title='BLEU vs F1 Score')

        # Save as interactive HTML
        pio.write_html(fig, os.path.join(output_dir, f'{lang_prefix}_interactive_scatter.html'))
    except ImportError:
        logger.info("Plotly not available, skipping interactive visualization")


# 2. Generate comprehensive HTML report
def generate_html_report(results, summary, output_dir, lang_prefix):
    """
    Generate a comprehensive HTML report with results and visualizations

    Args:
        results: List of evaluation results
        summary: Summary statistics
        output_dir: Directory to save the report
        lang_prefix: Language prefix for filenames
    """
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(results)

    # HTML template with Bootstrap for styling
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>QA Evaluation Report - {lang_prefix.upper()}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ padding: 20px; }}
            .metric-card {{ padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .chart-container {{ height: 300px; margin-bottom: 30px; }}
            .quality-excellent {{ background-color: #d5f5e3; }}
            .quality-good {{ background-color: #d6eaf8; }}
            .quality-partial {{ background-color: #fef9e7; }}
            .quality-poor {{ background-color: #fadbd8; }}
            .sample-container {{ margin-bottom: 20px; padding: 15px; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">QA Model Evaluation Report - {lang_prefix.upper()}</h1>

            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h3>Summary</h3>
                        </div>
                        <div class="card-body">
                            <p><strong>Total samples:</strong> {summary['total_samples']}</p>
                            <p><strong>Language:</strong> {summary['language']}</p>
                            <p><strong>Average generation time:</strong> {summary['average_metrics']['avg_generation_time']:.2f} seconds</p>

                            <h4 class="mt-4">Quality Distribution</h4>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <canvas id="qualityChart"></canvas>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>Quality</th>
                                                <th>Count</th>
                                                <th>Percentage</th>
                                            </tr>
                                        </thead>
                                        <tbody>
    """

    # Add quality distribution rows
    for quality, count in summary['quality_distribution'].items():
        percentage = count / summary['total_samples'] * 100
        html += f"""
                                            <tr>
                                                <td>{quality}</td>
                                                <td>{count}</td>
                                                <td>{percentage:.1f}%</td>
                                            </tr>
        """

    html += """
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            <h4 class="mt-4">Average Metrics</h4>
                            <div class="row">
    """

    # Add metric cards
    metrics = [
        ('BLEU', 'bleu', 'bg-info'),
        ('ROUGE-L', 'rougeL', 'bg-success'),
        ('F1 Score', 'f1_score', 'bg-warning'),
        ('Exact Match', 'exact_match', 'bg-danger')
    ]

    for metric_name, metric_key, bg_class in metrics:
        value = summary['average_metrics'].get(metric_key, 0)
        html += f"""
                                <div class="col-md-3">
                                    <div class="card metric-card {bg_class} text-white">
                                        <div class="card-body">
                                            <h5 class="card-title">{metric_name}</h5>
                                            <p class="card-text display-4">{value:.3f}</p>
                                        </div>
                                    </div>
                                </div>
        """

    html += """
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h3>Category Performance</h3>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="categoryChart"></canvas>
                            </div>

                            <table class="table table-striped mt-4">
                                <thead>
                                    <tr>
                                        <th>Category</th>
                                        <th>Sample Count</th>
                                        <th>BLEU</th>
                                        <th>ROUGE-L</th>
                                        <th>F1 Score</th>
                                        <th>Exact Match</th>
                                    </tr>
                                </thead>
                                <tbody>
    """

    # Add category rows
    for category, metrics in summary['category_metrics'].items():
        html += f"""
                                    <tr>
                                        <td>{category}</td>
                                        <td>{metrics['count']}</td>
                                        <td>{metrics.get('bleu', 0):.3f}</td>
                                        <td>{metrics.get('rougeL', 0):.3f}</td>
                                        <td>{metrics.get('f1_score', 0):.3f}</td>
                                        <td>{metrics.get('exact_match', 0):.3f}</td>
                                    </tr>
        """

    html += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header bg-warning text-dark">
                            <h3>Sample Results</h3>
                        </div>
                        <div class="card-body">
                            <div class="accordion" id="sampleAccordion">
    """

    # Add sample results (limit to 10 for each quality level)
    quality_samples = {
        'Excellent': df[df['quality'] == 'Excellent'].head(10).to_dict('records'),
        'Good': df[df['quality'] == 'Good'].head(10).to_dict('records'),
        'Partial': df[df['quality'] == 'Partial'].head(10).to_dict('records'),
        'Poor': df[df['quality'] == 'Poor'].head(10).to_dict('records')
    }

    for quality, samples in quality_samples.items():
        if not samples:
            continue

        html += f"""
                                <div class="card mb-3">
                                    <div class="card-header quality-{quality.lower()}">
                                        <h4>{quality} Samples ({len(samples)} shown)</h4>
                                    </div>
                                    <div class="card-body">
        """

        for i, sample in enumerate(samples):
            html += f"""
                                        <div class="sample-container quality-{quality.lower()}">
                                            <h5>Sample #{sample['id']}</h5>
                                            <p><strong>Question:</strong> {sample['question']}</p>
                                            <p><strong>Reference:</strong> {sample['reference_answer']}</p>
                                            <p><strong>Generated:</strong> {sample['generated_answer']}</p>
                                            <div class="row">
                                                <div class="col">BLEU: {sample['bleu']:.3f}</div>
                                                <div class="col">ROUGE-L: {sample['rougeL']:.3f}</div>
                                                <div class="col">F1: {sample['f1_score']:.3f}</div>
                                                <div class="col">EM: {sample['exact_match']:.3f}</div>
                                            </div>
                                        </div>
                                        <hr>
            """

        html += """
                                    </div>
                                </div>
        """

    html += """
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <footer class="mt-5 text-center text-muted">
                <p>QA Evaluation Report generated on {}</p>
            </footer>
        </div>

        <script>
            // Chart for quality distribution
            const qualityCtx = document.getElementById('qualityChart').getContext('2d');
            const qualityChart = new Chart(qualityCtx, {
                type: 'pie',
                data: {
                    labels: [{', '.join([f"'{q}'" for q in summary['quality_distribution'].keys()])}],
                    datasets: [{
                        data: [{', '.join([str(c) for c in summary['quality_distribution'].values()])}],
                        backgroundColor: ['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'right',
                        }
                    }
                }
            });

            // Chart for category performance
            const categoryCtx = document.getElementById('categoryChart').getContext('2d');
            const categoryChart = new Chart(categoryCtx, {
                type: 'bar',
                data: {
                    labels: [{', '.join([f"'{c}'" for c in summary['category_metrics'].keys()])}],
                    datasets: [
                        {
                            label: 'BLEU',
                            backgroundColor: 'rgba(54, 162, 235, 0.7)',
                            data: [{', '.join([str(m.get('bleu', 0)) for m in summary['category_metrics'].values()])}]
                        },
                        {
                            label: 'ROUGE-L',
                            backgroundColor: 'rgba(75, 192, 192, 0.7)',
                            data: [{', '.join([str(m.get('rougeL', 0)) for m in summary['category_metrics'].values()])}]
                        },
                        {
                            label: 'F1 Score',
                            backgroundColor: 'rgba(255, 206, 86, 0.7)',
                            data: [{', '.join([str(m.get('f1_score', 0)) for m in summary['category_metrics'].values()])}]
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """.format(time.strftime("%Y-%m-%d %H:%M:%S"))

    # Save the HTML report
    report_path = os.path.join(output_dir, f'{lang_prefix}_evaluation_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"HTML report generated and saved to {report_path}")

    return report_path


# 3. Generate comparative report for bilingual evaluation
def generate_comparative_report(en_results, ar_results, output_dir):
    """
    Generate a comparative HTML report for bilingual evaluation

    Args:
        en_results: English evaluation results
        ar_results: Arabic evaluation results
        output_dir: Directory to save the report
    """
    # Convert to DataFrame for easier processing
    en_df = pd.DataFrame(en_results)
    ar_df = pd.DataFrame(ar_results)

    # Calculate summary metrics
    en_metrics = {
        'bleu': en_df['bleu'].mean(),
        'rougeL': en_df['rougeL'].mean(),
        'f1_score': en_df['f1_score'].mean(),
        'exact_match': en_df['exact_match'].mean(),
        'sample_count': len(en_df)
    }

    ar_metrics = {
        'bleu': ar_df['bleu'].mean(),
        'rougeL': ar_df['rougeL'].mean(),
        'f1_score': ar_df['f1_score'].mean(),
        'exact_match': ar_df['exact_match'].mean(),
        'sample_count': len(ar_df)
    }

    # Quality distributions
    en_quality = en_df['quality'].value_counts().to_dict()
    ar_quality = ar_df['quality'].value_counts().to_dict()

    # HTML template with Bootstrap for styling
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bilingual QA Evaluation Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { padding: 20px; }
            .metric-card { padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .chart-container { height: 300px; margin-bottom: 30px; }
            .en-color { color: #3498db; }
            .ar-color { color: #e74c3c; }
            .comparison-table th, .comparison-table td { text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">Bilingual QA Evaluation Comparison</h1>

            <div class="row">
                <div class="col-md-12">
                    <div class="card mb-4">
                        <div class="card-header bg-dark text-white">
                            <h3>Overall Performance Comparison</h3>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <canvas id="metricsChart"></canvas>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <table class="table table-striped comparison-table">
                                        <thead>
                                            <tr>
                                                <th>Metric</th>
                                                <th class="en-color">English</th>
                                                <th class="ar-color">Arabic</th>
                                                <th>Difference</th>
                                            </tr>
                                        </thead>
                                        <tbody>
    """

    # Add metrics comparison rows
    metrics = [
        ('BLEU', 'bleu'),
        ('ROUGE-L', 'rougeL'),
        ('F1 Score', 'f1_score'),
        ('Exact Match', 'exact_match')
    ]

    for metric_name, metric_key in metrics:
        en_value = en_metrics[metric_key]
        ar_value = ar_metrics[metric_key]
        diff = en_value - ar_value
        diff_class = "text-success" if abs(diff) < 0.05 else ("text-danger" if diff > 0 else "text-warning")

        html += f"""
                                            <tr>
                                                <td>{metric_name}</td>
                                                <td class="en-color">{en_value:.3f}</td>
                                                <td class="ar-color">{ar_value:.3f}</td>
                                                <td class="{diff_class}">{diff:.3f}</td>
                                            </tr>
        """

    html += """
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-12">
                    <div class="card mb-4">
                        <div class="card-header bg-primary text-white">
                            <h3>Quality Distribution Comparison</h3>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="qualityChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card mb-4">
                        <div class="card-header bg-info text-white">
                            <h3>English Samples (10 Random)</h3>
                        </div>
                        <div class="card-body" style="max-height: 600px; overflow-y: auto;">
    """

    # Add 10 random English samples
    en_samples = en_df.sample(min(10, len(en_df))).to_dict('records')
    for sample in en_samples:
        html += f"""
                            <div class="card mb-3">
                                <div class="card-header">
                                    <strong>Quality: {sample['quality']}</strong> 
                                    <span class="float-end">
                                        BLEU: {sample['bleu']:.2f} | F1: {sample['f1_score']:.2f}
                                    </span>
                                </div>
                                <div class="card-body">
                                    <p><strong>Question:</strong> {sample['question']}</p>
                                    <p><strong>Reference:</strong> {sample['reference_answer']}</p>
                                    <p><strong>Generated:</strong> {sample['generated_answer']}</p>
                                </div>
                            </div>
        """

    html += """
                        </div>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="card mb-4">
                        <div class="card-header bg-danger text-white">
                            <h3>Arabic Samples (10 Random)</h3>
                        </div>
                        <div class="card-body" style="max-height: 600px; overflow-y: auto;">
    """

    # Add 10 random Arabic samples
    ar_samples = ar_df.sample(min(10, len(ar_df))).to_dict('records')
    for sample in ar_samples:
        html += f"""
                            <div class="card mb-3">
                                <div class="card-header">
                                    <strong>Quality: {sample['quality']}</strong> 
                                    <span class="float-end">
                                        BLEU: {sample['bleu']:.2f} | F1: {sample['f1_score']:.2f}
                                    </span>
                                </div>
                                <div class="card-body">
                                    <p><strong>Question:</strong> {sample['question']}</p>
                                    <p><strong>Reference:</strong> {sample['reference_answer']}</p>
                                    <p><strong>Generated:</strong> {sample['generated_answer']}</p>
                                </div>
                            </div>
        """

    html += """
                        </div>
                    </div>
                </div>
            </div>

            <footer class="mt-5 text-center text-muted">
                <p>Bilingual QA Evaluation Report generated on {}</p>
            </footer>
        </div>

        <script>
            // Chart for metrics comparison
            const metricsCtx = document.getElementById('metricsChart').getContext('2d');
            const metricsChart = new Chart(metricsCtx, {
                type: 'radar',
                data: {
                    labels: ['BLEU', 'ROUGE-L', 'F1 Score', 'Exact Match'],
                    datasets: [
                        {
                            label: 'English',
                            data: [{en_metrics['bleu']}, {en_metrics['rougeL']}, {en_metrics['f1_score']}, {en_metrics['exact_match']}],
                            backgroundColor: 'rgba(52, 152, 219, 0.2)',
                            borderColor: 'rgba(52, 152, 219, 1)',
                            pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                        },
                        {
                            label: 'Arabic',
                            data: [{ar_metrics['bleu']}, {ar_metrics['rougeL']}, {ar_metrics['f1_score']}, {ar_metrics['exact_match']}],
                            backgroundColor: 'rgba(231, 76, 60, 0.2)',
                            borderColor: 'rgba(231, 76, 60, 1)',
                            pointBackgroundColor: 'rgba(231, 76, 60, 1)',
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        r: {
                            min: 0,
                            max: 1,
                            ticks: {
                                stepSize: 0.2
                            }
                        }
                    }
                }
            });

            // Chart for quality distribution
            const qualityCtx = document.getElementById('qualityChart').getContext('2d');
            const qualityLabels = Array.from(new Set([...Object.keys({en_quality}), ...Object.keys({ar_quality})]));

            const qualityChart = new Chart(qualityCtx, {
                type: 'bar',
                data: {
                    labels: qualityLabels,
                    datasets: [
                        {
                            label: 'English',
                            backgroundColor: 'rgba(52, 152, 219, 0.7)',
                            data: qualityLabels.map(q => ({en_quality}[q] || 0))
                        },
                        {
                            label: 'Arabic',
                            backgroundColor: 'rgba(231, 76, 60, 0.7)',
                            data: qualityLabels.map(q => ({ar_quality}[q] || 0))
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """.format(time.strftime("%Y-%m-%d %H:%M:%S"))

    # Save the HTML report
    report_path = os.path.join(output_dir, 'bilingual_comparison_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Bilingual comparison report generated and saved to {report_path}")

    return report_path


def prepare_human_evaluation_sheet(results, output_path, sample_size=50):
    """
    Create a spreadsheet for human evaluators to rate Arabic answers

    Args:
        results: List of evaluation results
        output_path: Path to save the evaluation sheet
        sample_size: Number of samples to include in the evaluation
    """
    try:
        import random

        # Select a subset of results for human evaluation
        evaluation_size = min(sample_size, len(results))
        human_eval_samples = random.sample(results, evaluation_size)

        # Create a DataFrame
        df = pd.DataFrame({
            'id': [sample['id'] for sample in human_eval_samples],
            'question': [sample['question'] for sample in human_eval_samples],
            'reference_answer': [sample['reference_answer'] for sample in human_eval_samples],
            'generated_answer': [sample['generated_answer'] for sample in human_eval_samples],
            'auto_metrics': [
                f"BLEU={sample.get('bleu', 0):.2f}, F1={sample.get('f1_score', 0):.2f}, EM={sample.get('exact_match', 0):.2f}"
                for sample in human_eval_samples],
            'fluency_score (1-5)': '',  # Scale from 1-5
            'accuracy_score (1-5)': '',  # Scale from 1-5
            'completeness_score (1-5)': '',  # Scale from 1-5
            'evaluator_notes': ''
        })

        # Save as Excel file
        df.to_excel(output_path, index=False)
        logger.info(f"Human evaluation sheet prepared at {output_path} with {evaluation_size} samples")

        # Create evaluation guidelines
        guidelines_path = os.path.join(os.path.dirname(output_path), "human_evaluation_guidelines.txt")
        with open(guidelines_path, 'w', encoding='utf-8') as f:
            f.write("""
            HUMAN EVALUATION GUIDELINES
            --------------------------

            Please rate each generated answer on the following criteria:

            1. FLUENCY (1-5)
               - How natural and fluent is the Arabic text?
               - 1: Completely unnatural or incomprehensible
               - 3: Acceptable but with minor issues
               - 5: Perfect, natural Arabic

            2. ACCURACY (1-5)
               - How factually accurate is the answer compared to reference?
               - 1: Completely incorrect
               - 3: Partially correct with some errors
               - 5: Completely accurate

            3. COMPLETENESS (1-5)
               - Does the answer provide all the necessary information?
               - 1: Missing critical information
               - 3: Contains main points but lacks some details
               - 5: Complete answer with all necessary details

            Additional notes:
            - Please add any observations about particular error patterns
            - Note any cases where the automatic metrics seem to be wrong
            - Identify any dialectal or cultural nuances that might be missed by automatic evaluation
            """)
    except Exception as e:
        logger.error(f"Error preparing human evaluation sheet: {e}")
