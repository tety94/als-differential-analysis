import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from utilities.utils import save_plot
from config import min_common_models


def plot_comparisons(res_df, folder):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=res_df.index, y='f1', data=res_df.sort_values('f1', ascending=False), ax=ax)
    plt.xticks(rotation=45);
    plt.title('F1 score medio per modello');
    plt.tight_layout()
    save_plot(fig, os.path.join(folder, 'f1_comparison.png'))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=res_df.index, y='accuracy', data=res_df.sort_values('accuracy', ascending=False), ax=ax)
    plt.xticks(rotation=45);
    plt.title('Accuracy media per modello');
    plt.tight_layout()
    save_plot(fig, os.path.join(folder, 'accuracy_comparison.png'))


def common_features_plot(model_feature_importances, folder):
    top_features_sets = [set(fi.index) for fi in model_feature_importances.values()]
    all_features = [f for s in top_features_sets for f in s]
    common_counts = Counter(all_features)
    common_features = [f for f, c in common_counts.items() if c >= min_common_models]

    if common_features:
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = [common_counts[f] for f in common_features]
        sns.barplot(x=counts, y=common_features, ax=ax)
        plt.xlabel('Numero di modelli');
        plt.title(f'Feature comuni in almeno {min_common_models} modelli')
        plt.tight_layout()
        save_plot(fig, os.path.join(folder, 'common_features.png'))

    return common_features
