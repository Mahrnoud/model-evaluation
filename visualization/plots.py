def create_visualizations(results, evaluation_metrics, category_metrics, output_dir, lang_prefix):
    """
    Create visualizations from evaluation results with enhanced Arabic support

    Args:
        results: List of evaluation results
        evaluation_metrics: Dictionary with evaluation metrics
        category_metrics: Dictionary with category-wise metrics
        output_dir: Directory to save visualizations
        lang_prefix: Language prefix for filenames
    """
    # Plot metrics distribution
    plt.figure(figsize=(15, 10))

    # Create a 2x2 grid of plots for different metrics
    plt.subplot(2, 2, 1)
    plt.hist(evaluation_metrics['bleu'], bins=20, alpha=0.7)
    plt.axvline(np.mean(evaluation_metrics['bleu']), color='red', linestyle='dashed', linewidth=1)
    plt.title('BLEU Score Distribution')
    plt.xlabel('BLEU Score')
    plt.ylabel('Count')

    plt.subplot(2, 2, 2)
    plt.hist(evaluation_metrics['rougeL'], bins=20, alpha=0.7)
    plt.axvline(np.mean(evaluation_metrics['rougeL']), color='red', linestyle='dashed', linewidth=1)
    plt.title('ROUGE-L Score Distribution')
    plt.xlabel('ROUGE-L Score')
    plt.ylabel('Count')

    plt.subplot(2, 2, 3)
    plt.hist(evaluation_metrics['f1_score'], bins=20, alpha=0.7)
    plt.axvline(np.mean(evaluation_metrics['f1_score']), color='red', linestyle='dashed', linewidth=1)
    plt.title('F1 Word Match Distribution')
    plt.xlabel('F1 Score')
    plt.ylabel('Count')

    plt.subplot(2, 2, 4)
    # Get exact match ratio
    exact_match_ratio = sum(1 for em in evaluation_metrics['exact_match'] if em > 0.9) / len(
        evaluation_metrics['exact_match'])
    plt.bar(['Not Exact Match', 'Exact Match'], [1 - exact_match_ratio, exact_match_ratio])
    plt.title('Exact Match Ratio')
    plt.ylabel('Proportion')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lang_prefix}_score_distributions.png'))
    plt.close()

    # Plot quality distribution
    quality_counts = {
        'Excellent': len([r for r in results if r['quality'] == 'Excellent']),
        'Good': len([r for r in results if r['quality'] == 'Good']),
        'Partial': len([r for r in results if r['quality'] == 'Partial']),
        'Poor': len([r for r in results if r['quality'] == 'Poor'])
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(quality_counts.keys(), quality_counts.values())

    # Add percentage labels
    total = sum(quality_counts.values())
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height / total:.1%}',
                 ha='center', va='bottom')

    plt.title('Quality Distribution of Generated Answers')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'{lang_prefix}_quality_distribution.png'))
    plt.close()

    # Plot category-wise performance if categories exist
    if category_metrics:
        categories = list(category_metrics.keys())

        # For cleaner display, limit to top N categories if there are many
        if len(categories) > 10:
            # Sort categories by count
            categories = sorted(categories,
                                key=lambda x: category_metrics[x].get('count', 0),
                                reverse=True)[:10]

        bleu_scores = [category_metrics[cat]['bleu'] for cat in categories]
        rouge_scores = [category_metrics[cat]['rougeL'] for cat in categories]
        f1_scores = [category_metrics[cat].get('f1_score', 0) for cat in categories]

        plt.figure(figsize=(14, 7))

        x = np.arange(len(categories))
        width = 0.25

        plt.bar(x - width, bleu_scores, width, label='BLEU', color='blue', alpha=0.7)
        plt.bar(x, rouge_scores, width, label='ROUGE-L', color='green', alpha=0.7)
        plt.bar(x + width, f1_scores, width, label='F1', color='orange', alpha=0.7)

        plt.xlabel('Category')
        plt.ylabel('Score')
        plt.title('QA Performance by Category')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{lang_prefix}_category_performance.png'))
        plt.close()

    # Plot generation time histogram
    plt.figure(figsize=(10, 6))
    plt.hist(evaluation_metrics['generation_time'], bins=20, alpha=0.7)
    plt.axvline(np.mean(evaluation_metrics['generation_time']), color='red', linestyle='dashed', linewidth=1)
    plt.title('Answer Generation Time Distribution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lang_prefix}_generation_time.png'))
    plt.close()

    # Plot error analysis - Questions with lowest scores
    n_worst = min(10, len(results))
    sorted_results = sorted(results, key=lambda x: x['bleu'])[:n_worst]

    worst_questions = [r['question'][:40] + '...' if len(r['question']) > 40 else r['question'] for r in sorted_results]
    worst_scores = [r['bleu'] for r in sorted_results]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(worst_questions)), worst_scores, color='salmon')
    plt.yticks(range(len(worst_questions)), worst_questions)
    plt.xlabel('BLEU Score')
    plt.title('Questions with Lowest Performance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lang_prefix}_worst_questions.png'))
    plt.close()

    # Plot metric correlation
    if len(results) > 5:  # Only if we have enough samples
        plt.figure(figsize=(14, 10))

        # BLEU vs ROUGE-L
        plt.subplot(2, 2, 1)
        plt.scatter(
            [r['bleu'] for r in results],
            [r['rougeL'] for r in results],
            alpha=0.5
        )
        plt.xlabel('BLEU Score')
        plt.ylabel('ROUGE-L Score')
        plt.title('BLEU vs ROUGE-L Correlation')

        # BLEU vs F1
        plt.subplot(2, 2, 2)
        plt.scatter(
            [r['bleu'] for r in results],
            [r['f1_score'] for r in results],
            alpha=0.5
        )
        plt.xlabel('BLEU Score')
        plt.ylabel('F1 Score')
        plt.title('BLEU vs F1 Correlation')

        # ROUGE-L vs F1
        plt.subplot(2, 2, 3)
        plt.scatter(
            [r['rougeL'] for r in results],
            [r['f1_score'] for r in results],
            alpha=0.5
        )
        plt.xlabel('ROUGE-L Score')
        plt.ylabel('F1 Score')
        plt.title('ROUGE-L vs F1 Correlation')

        # Exact Match vs F1
        plt.subplot(2, 2, 4)
        plt.scatter(
            [r['exact_match'] for r in results],
            [r['f1_score'] for r in results],
            alpha=0.5
        )
        plt.xlabel('Exact Match Score')
        plt.ylabel('F1 Score')
        plt.title('Exact Match vs F1 Correlation')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{lang_prefix}_metric_correlations.png'))
        plt.close()

        # Generate a plot showing metric agreement/disagreement
        plt.figure(figsize=(10, 6))
        agree_count = sum(1 for r in results if abs(r['bleu'] - r['f1_score']) < 0.2)
        disagree_count = len(results) - agree_count

        plt.bar(['Metrics Agree', 'Metrics Disagree'], [agree_count, disagree_count])
        plt.title('Metric Agreement (BLEU vs F1)')
        plt.ylabel('Count')

        # Add percentage labels
        plt.text(0, agree_count + 1, f'{agree_count / len(results):.1%}', ha='center')
        plt.text(1, disagree_count + 1, f'{disagree_count / len(results):.1%}', ha='center')

        plt.savefig(os.path.join(output_dir, f'{lang_prefix}_metric_agreement.png'))
        plt.close()

def create_comparative_visualization(results, output_dir):
    """
    Create visualizations comparing QA performance across languages with enhanced Arabic support

    Args:
        results: Dictionary with evaluation results for different languages
        output_dir: Directory to save visualizations
    """
    if not results or len(results) < 2:
        return

    # Extract metrics for comparison
    languages = list(results.keys())

    # Basic metrics
    metrics_to_compare = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'f1_score', 'exact_match']
    extracted_metrics = {}

    for metric in metrics_to_compare:
        extracted_metrics[metric] = [
            results[lang]['average_metrics'].get(metric, 0)
            for lang in languages
        ]

    # Add generation time
    times = [results[lang]['average_metrics'].get('avg_generation_time', 0) for lang in languages]

    # Plot comparison of metrics
    plt.figure(figsize=(12, 8))

    x = np.arange(len(metrics_to_compare))
    width = 0.35

    values_lang1 = [extracted_metrics[metric][0] for metric in metrics_to_compare]
    values_lang2 = [extracted_metrics[metric][1] for metric in metrics_to_compare]

    plt.bar(x - width / 2, values_lang1, width, label=f'{languages[0].upper()} Questions')
    plt.bar(x + width / 2, values_lang2, width, label=f'{languages[1].upper()} Questions')

    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('QA Performance Comparison by Language')
    plt.xticks(x, [m.upper() for m in metrics_to_compare])
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_comparison.png'))
    plt.close()

    # Plot generation time comparison
    plt.figure(figsize=(8, 6))
    plt.bar(languages, times, alpha=0.7)
    plt.xlabel('Question Language')
    plt.ylabel('Average Time (seconds)')
    plt.title('Answer Generation Time by Language')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_generation_time.png'))
    plt.close()

    # If there are common categories across languages, plot category comparison
    if all('category_metrics' in results[lang] for lang in languages):
        # Find common categories
        categories_lang1 = set(results[languages[0]]['category_metrics'].keys())
        categories_lang2 = set(results[languages[1]]['category_metrics'].keys())
        common_categories = categories_lang1.intersection(categories_lang2)

        if common_categories:
            # Limit to top N categories if there are many
            if len(common_categories) > 8:
                # For each category, compute the average of BLEU scores across both languages
                category_scores = {}
                for cat in common_categories:
                    avg_score = (
                                        results[languages[0]]['category_metrics'][cat]['bleu'] +
                                        results[languages[1]]['category_metrics'][cat]['bleu']
                                ) / 2
                    category_scores[cat] = avg_score

                # Keep only the top 8 categories by average score
                common_categories = sorted(
                    category_scores.keys(),
                    key=lambda cat: category_scores[cat],
                    reverse=True
                )[:8]

            # Compare performance across languages by category
            plt.figure(figsize=(14, 8))

            x = np.arange(len(common_categories))
            width = 0.35

            bleu_lang1 = [results[languages[0]]['category_metrics'][cat]['bleu'] for cat in common_categories]
            bleu_lang2 = [results[languages[1]]['category_metrics'][cat]['bleu'] for cat in common_categories]

            plt.bar(x - width / 2, bleu_lang1, width, label=f'{languages[0].upper()} Questions')
            plt.bar(x + width / 2, bleu_lang2, width, label=f'{languages[1].upper()} Questions')

            plt.xlabel('Category')
            plt.ylabel('BLEU Score')
            plt.title('QA Performance Comparison by Category and Language')
            plt.xticks(x, common_categories, rotation=45, ha='right')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'category_language_comparison.png'))
            plt.close()

            # Also create a radar chart for multi-metric comparison if matplotlib version supports it
            try:
                # Calculate mean scores by category for both languages
                categories = list(common_categories)

                # Get metrics for first language
                metrics_lang1 = {
                    'bleu': [results[languages[0]]['category_metrics'][cat]['bleu'] for cat in categories],
                    'rougeL': [results[languages[0]]['category_metrics'][cat]['rougeL'] for cat in categories],
                    'f1_score': [results[languages[0]]['category_metrics'][cat].get('f1_score', 0) for cat in
                                 categories]
                }

                # Get metrics for second language
                metrics_lang2 = {
                    'bleu': [results[languages[1]]['category_metrics'][cat]['bleu'] for cat in categories],
                    'rougeL': [results[languages[1]]['category_metrics'][cat]['rougeL'] for cat in categories],
                    'f1_score': [results[languages[1]]['category_metrics'][cat].get('f1_score', 0) for cat in
                                 categories]
                }

                # Create a radar chart for comparing languages across categories
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, polar=True)

                # Number of categories
                N = len(categories)

                # Angles for the radar chart
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                angles += angles[:1]  # Close the loop

                # Add category labels
                plt.xticks(angles[:-1], categories, size=8)

                # Plot data for first language
                values = metrics_lang1['bleu']
                values += values[:1]  # Close the loop
                ax.plot(angles, values, 'o-', linewidth=2, label=f'{languages[0].upper()} BLEU')
                ax.fill(angles, values, alpha=0.1)

                # Plot data for second language
                values = metrics_lang2['bleu']
                values += values[:1]  # Close the loop
                ax.plot(angles, values, 'o-', linewidth=2, label=f'{languages[1].upper()} BLEU')
                ax.fill(angles, values, alpha=0.1)

                # Add legend and title
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.title(f'BLEU Score Comparison by Category and Language', size=11, y=1.1)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'radar_category_comparison.png'))
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create radar chart: {e}")

