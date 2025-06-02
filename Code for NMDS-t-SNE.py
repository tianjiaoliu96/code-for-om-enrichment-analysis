# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:08:38 2025

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from skopt import gp_minimize
from skopt.space import Integer
import warnings

# Ignore all warnings.
warnings.filterwarnings("ignore")

# Data input
data1 = pd.read_excel('S6 terrigenous clastic input dataset for input.xlsx', index_col=0)
data = data1.iloc[:, :-2]  # Exclude the final two columns (TOC, PG).
parameters = list(data.columns)
regions = data1.index

# Define a function to replace outliers that exceed the upper and lower limits with sub-outliers
def replace_outliers(series):
    Q1 = series.quantile(0.15)
    Q3 = series.quantile(0.85)
    IQR = Q3 - Q1
    lower_boundary = Q1 - 2.0 * IQR
    upper_boundary = Q3 + 2.0 * IQR
    outliers_mask = ~((series >= lower_boundary) & (series <= upper_boundary))
    
    # For each outlier, find the nearest non-outliers and replace with their mean
    for idx in series[outliers_mask].index:
        # Find previous non-outlier
        prev_idx = int(idx) - 1 if idx.isdigit() and int(idx) > 0 else None
        
        # Find next non-outlier
        next_idx = int(idx) + 1 if idx.isdigit() and int(idx) < len(series) else None
        
        # Calculate mean of the nearest non-outliers
        if prev_idx is not None and next_idx is not None:
            series[idx] = (series[prev_idx] + series[next_idx]) / 2
        elif prev_idx is not None:
            series[idx] = series[prev_idx]
        elif next_idx is not None:
            series[idx] = series[next_idx]
    
    return series

# Apply outlier treatment to each parameter. 
for param in parameters:
    data[param] = replace_outliers(data[param])

# Standardize using Z-score. 
scaler = StandardScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

def calculate_r2(observed, predicted):
    total_sum_squares = np.sum((observed - np.mean(observed)) ** 2)
    residual_sum_squares = np.sum((observed - predicted) ** 2)
    return 1 - (residual_sum_squares / total_sum_squares)

def evaluate_parameters(parameters, data):
    subset_data = data[parameters]
    dist_matrix = squareform(pdist(subset_data.values))
    nmds = MDS(n_components=2, dissimilarity='precomputed', max_iter=5000, eps=1e-3, n_init=100, metric=False, random_state=42, n_jobs=-1)
    nmds_result = nmds.fit_transform(dist_matrix)
    stress = nmds.stress_
    ordination_dist_matrix = squareform(pdist(nmds_result))
    ordination_distances = ordination_dist_matrix[0]
    observed_dissimilarity = dist_matrix[0]
    non_metric_r2 = calculate_r2(observed_dissimilarity, ordination_distances)
    linear_fit_r2 = calculate_r2(np.polyval(np.polyfit(observed_dissimilarity, ordination_distances, 1), observed_dissimilarity), ordination_distances)
    return stress, nmds_result, observed_dissimilarity, non_metric_r2, linear_fit_r2

def evaluate(parameters):
    stress, _, observed_dissimilarity, non_metric_r2, linear_fit_r2 = evaluate_parameters(parameters, normalized_data)
    fitness = -0.1 * stress - 0.2 * np.sum(observed_dissimilarity) + 0.3 * non_metric_r2 + 0.4 * linear_fit_r2
    return fitness

# Implement Bayesian optimization function.
def optimize_bayes():

    history = [] 
    
    def objective(params):
        subset = [parameters[i] for i in range(len(params)) if params[i]]
        
        if not subset:  # If the subset is empty
    # Record historical data (maintain consistent format)
          history.append({
        'n_valid': 0,
        'fitness': 1e10,  # Large penalty value
        'params': params
        })
        return 1e10  # Return a large positive value indicating an invalid solution
# ========Modification End ========
        fitness = -evaluate(subset)
        
        # Record data for each iteration
        history.append({
            'n_valid': len(subset),
            'fitness': fitness,
            'params': params
        })
        
        return fitness

    space = [Integer(0, 1) for _ in range(len(parameters))]
    print("Space dimensions:", len(space))
    res = gp_minimize(objective, space, n_calls=60, random_state=42, verbose=True)
    best_params = [parameters[i] for i in range(len(res.x)) if res.x[i]]
    return res, best_params, history 
res, best_parameters, opt_history = optimize_bayes() 

# Plot the relationship between fitness and the number of valid parameters
plt.figure(figsize=(10, 6))
plt.scatter(
    x=[h['n_valid'] for h in opt_history],
    y=[h['fitness'] for h in opt_history],
    c='steelblue',
    alpha=0.7,
    edgecolors='w',
    s=80
)
plt.title('Fitness vs. Number of Valid Parameters', fontsize=14)
plt.xlabel('Number of Valid Parameters', fontsize=12)
plt.ylabel('Fitness Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# ================== MDS Visualization of Parameter Validity ==================
# Generate a similarity matrix between parameters (using correlation coefficients)
param_corr = np.abs(normalized_data.corr().values)
param_dist = 1 - param_corr  # Convert correlation coefficients into distances

# Visualize the parameter space using MDS
param_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
param_mds_result = param_mds.fit_transform(param_dist)

# Mark parameter validity (based on Bayesian optimization results)
param_status = ['valid' if param in best_parameters else 'invalid' for param in parameters]

# Plot parameter MDS visualization
plt.figure(figsize=(10, 8))

# Plot valid parameters (blue dots)
valid_mask = np.array(param_status) == 'valid'
plt.scatter(
    param_mds_result[valid_mask, 0],
    param_mds_result[valid_mask, 1],
    c='royalblue',
    marker='o',
    s=120,
    edgecolor='w',
    linewidth=1.5,
    label='Valid Parameters'
)

# Plot invalid parameters (red crosses)
invalid_mask = np.array(param_status) == 'invalid'
plt.scatter(
    param_mds_result[invalid_mask, 0],
    param_mds_result[invalid_mask, 1],
    c='crimson',
    marker='x',
    s=120,
    linewidth=2.5,
    label='Invalid Parameters'
)

# Add parameter labels
for i, (x, y) in enumerate(param_mds_result):
    plt.text(
        x + 0.02,
        y + 0.02,
        parameters[i],
        fontsize=10,
        alpha=0.8,
        weight='bold'
    )

plt.title('Parameter Effectiveness in MDS Space (Feature-level)', fontsize=14)
plt.xlabel('MDS Dimension 1', fontsize=12)
plt.ylabel('MDS Dimension 2', fontsize=12)
plt.legend(title="Parameter Status", loc='best')
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()


print("Best parameters:", best_parameters)
stress, best_nmds_result, best_observed_dissimilarity, best_non_metric_r2, best_linear_fit_r2 = evaluate_parameters(best_parameters, normalized_data)

print("Best stress:", stress)
print("Non-metric R2:", best_non_metric_r2)
print("Linear fit R2:", best_linear_fit_r2)

#  Plot scatter diagrams.
plt.scatter(best_observed_dissimilarity.flatten(), best_nmds_result[:, 0], c='b', label='Observations')
plt.text(0.5, 0.9, f"Non-metric R2: {best_non_metric_r2:.2f}", transform=plt.gca().transAxes)
plt.text(0.5, 0.85, f"Linear fit R2: {best_linear_fit_r2:.2f}", transform=plt.gca().transAxes)
plt.xlabel('Ordination Distances')
plt.ylabel('Observed Dissimilarity')
plt.title('Correlation between Observed Dissimilarity and Ordination Distances')
plt.legend()
plt.show()

# Add noise to the data.
def add_noise(data, regions, intra_region_noise=0.01, inter_region_noise=0.02):
    noisy_data = data.copy()
    for region in np.unique(regions):
        region_data = noisy_data[regions == region]
        noise = np.random.normal(scale=intra_region_noise, size=region_data.shape)
        noisy_data[regions == region] += noise
    
    global_noise = np.random.normal(scale=inter_region_noise, size=data.shape)
    noisy_data += global_noise
    return noisy_data

from sklearn.cluster import KMeans

# Calculate three scoring metrics and a composite score.
def calculate_scores(labels, data_scaled):
    silhouette = silhouette_score(data_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(data_scaled, labels)
    davies_bouldin = davies_bouldin_score(data_scaled, labels)
    composite = (0.4 * silhouette) + (0.4 * calinski_harabasz) - (0.2 * davies_bouldin)
    return silhouette, calinski_harabasz, davies_bouldin, composite

# Optimize t-SNE parameters. 
def optimize_tsne(data, regions, max_iterations=200, convergence_threshold=1e-4):
    """
    Optimize t-SNE parameters to obtain the best composite score and distance score.
    :param data: Input data
    :param regions: Region labels
    :param max_iterations: Maximum number of iterations, default is 200
    :param convergence_threshold: Convergence threshold, default is 1e-4
    :return: Best t-SNE parameters, best t-SNE result, best evaluation metric tuple, and best composite score
    """
    best_composite = float('-inf')  # Primary objective: composite score (maximize first)
    best_distance = float('-inf')  # Secondary objective: distance score (maximize when primary objective is the same)
    best_tsne_params = None
    best_tsne_result = None
    best_scores = (0, 0, 0)  # Keep the original scoring format: (silhouette, calinski, davies)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    true_regions = np.array(regions)  # True region labels (for distance calculation)

    for perp in np.linspace(5, 50, 12, dtype=int):
        for lr in np.linspace(200, 2000, 23, dtype=int):
            for rs in np.linspace(0, 100, 11, dtype=int):
                tsne = TSNE(n_components=2, perplexity=perp, learning_rate=lr, random_state=int(rs))
                try:
                    tsne_result = tsne.fit_transform(data_scaled)
                except (ValueError, TypeError):
                    continue  # Skip parameter combinations that fail to calculate

                kmeans = KMeans(n_clusters=4, random_state=int(rs))
                cluster_labels = kmeans.fit_predict(tsne_result)
                sil, cal, dav, comp = calculate_scores(cluster_labels, data_scaled)
                current_composite = comp
                current_distance = compute_distance_value(tsne_result, true_regions)

                if current_composite > best_composite:
                    # # When the current composite score is greater than the best composite score, update all best information
                    best_composite = current_composite
                    best_distance = current_distance
                    best_tsne_params = (perp, lr, int(rs))
                    best_tsne_result = tsne_result
                    best_scores = (sil, cal, dav)
                elif current_composite == best_composite:
                    if current_distance > best_distance:
                        # When the composite scores are the same, but the current distance score is larger, update the distance score and related information
                        best_distance = current_distance
                        best_tsne_params = (perp, lr, int(rs))
                        best_tsne_result = tsne_result
                        best_scores = (sil, cal, dav)

    return best_tsne_params, best_tsne_result, best_scores, best_composite

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA dimensionality reduction.
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# Execute k-means clustering. 
kmeans = KMeans(n_clusters=4)
cluster_labels = kmeans.fit_predict(pca_result)

# Compute the scoring metrics.
pca_silhouette = silhouette_score(data_scaled, cluster_labels)
pca_calinski_harabasz = calinski_harabasz_score(data_scaled, cluster_labels)
pca_davies_bouldin = davies_bouldin_score(data_scaled, cluster_labels)

# Print the scoring metrics for PCA.
print(f"PCA Silhouette Score: {pca_silhouette}")
print(f"PCA Calinski-Harabasz Score: {pca_calinski_harabasz}")
print(f"PCA Davies-Bouldin Score: {pca_davies_bouldin}")


# Compute distance value
def compute_distance_value(data, regions, metric='euclidean'):
    unique_regions = np.unique(regions)
    n_clusters = len(unique_regions)

    intra_cluster_distances = np.mean([np.mean(pdist(data[regions == region], metric=metric)) for region in unique_regions])

    inter_cluster_distances = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            inter_cluster_distances += np.mean(cdist(data[regions == unique_regions[i]], data[regions == unique_regions[j]], metric=metric))
    inter_cluster_distances /= (n_clusters * (n_clusters - 1) / 2)

    return inter_cluster_distances - intra_cluster_distances

# Plot t-SNE scatter plot with best parameters
valid_params = list(best_parameters)
valid_data = normalized_data[valid_params]
best_tsne_params, tsne_result, best_scores, best_score = optimize_tsne(valid_data, regions)
# Print the scoring metrics for t-sne
tsne_silhouette, tsne_calinski_harabasz, tsne_davies_bouldin = best_scores
print(f"t-SNE Silhouette Score: {tsne_silhouette}")
print(f"t-SNE Calinski-Harabasz Score: {tsne_calinski_harabasz}")
print(f"t-SNE Davies-Bouldin Score: {tsne_davies_bouldin}")
print(f"Composite Score: {best_score}")
region_mapping = {
    'DHS': ('red', 'o'),
    'GDK': ('blue', 'x'),
    'HYC': ('green', 's'),
    'JJZG': ('black', '^'),
}

legend_elements = [patches.Patch(facecolor=value[0], edgecolor='black', label=key) for key, value in region_mapping.items()]

plt.subplot(1, 2, 2)
for region, coords in zip(regions, tsne_result):
    color, marker = region_mapping.get(region, ('gray', 'o'))
    plt.scatter(coords[0], coords[1], color=color, marker=marker)
plt.title(f"t-SNE for Regions (perplexity={best_tsne_params[0]}, learning_rate={best_tsne_params[1]})")

# --------------------------- Key adjustment: Automatic optimization of legend position ---------------------------
plt.legend(
    handles=legend_elements, 
    loc='best',  #  # Automatically select the position that least obscures the data (inside the figure frame)
    frameon=True,  # Keep the legend border (optional, original logic unchanged)
    framealpha=0.9  # Semi-transparent background to reduce the sense of occlusion (optional, original logic unchanged)
)

plt.tight_layout()
plt.show()
# ================== Parameter Importance Visualization ==================
param_importance = {}
for param in parameters:
    param_values = normalized_data[param].values
    corr_dim1 = np.abs(np.corrcoef(param_values, best_nmds_result[:, 0])[0, 1])
    corr_dim2 = np.abs(np.corrcoef(param_values, best_nmds_result[:, 1])[0, 1])
    param_importance[param] = (corr_dim1 + corr_dim2) / 2

sorted_importance = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp2d

# ================== Contour Plot Parameter Selection==================
# Get the top two most important parameters
top_params = [sorted_importance[0][0], sorted_importance[1][0]]
print(f"\nMost important parameters: {top_params}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, LinearNDInterpolator
import pandas as pd
from pathlib import Path

# Define region mapping (keep the original logic)
region_mapping = {
    'DHS': ('red', 'o'),
    'GDK': ('blue', 'x'),
    'HYC': ('green', 's'),
    'JJZG': ('black', '^'),
}

def plot_contour(tsne_result, regions, selected_param, values, region_mapping, excel_writer):
    # ==== New: Extraction of measured point data ====
    # Ensure the data lengths are consistent (key check)
    if selected_param in data1.columns:
        raw_values = data1[selected_param].values  # Use the original data values
    else:
        raise ValueError(f"Parameter {selected_param} does not exist in the original data")

    # ==== Extraction of measured point data ====
    measured_points = {
        'X Coordinate': tsne_result[:, 0],
        'Y Coordinate': tsne_result[:, 1],
        'Z Value ({})'.format(selected_param): raw_values  # Use the original data values
    }
    
    # Create a DataFrame and write it to Excel
    df = pd.DataFrame(measured_points)

    # Process special characters in the worksheet name (Excel restrictions)
    # Remove disallowed characters
    sheet_name = selected_param
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        sheet_name = sheet_name.replace(char, '')
    sheet_name = sheet_name[:31]  # Truncate parameter names longer than 31 characters
    df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
    
    # ==== Keep the original plotting logic unchanged ====
    values = np.clip(values, 0, None)
    vmin = 0
    vmax = values.max() if values.size > 0 else 0
    
    tsne_x, tsne_y = tsne_result[:, 0], tsne_result[:, 1]
    grid_x, grid_y = np.mgrid[tsne_x.min():tsne_x.max():200j, tsne_y.min():tsne_y.max():200j]
    
    grid_values = griddata(tsne_result, values, (grid_x, grid_y), method='cubic')
    grid_values = np.clip(grid_values, 0, None)
    
    if np.isnan(grid_values).any():
        grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
        non_nan_mask = ~np.isnan(grid_values.flatten())
        f = LinearNDInterpolator(grid_points[non_nan_mask], grid_values.flatten()[non_nan_mask])
        grid_values = f(grid_points).reshape(grid_x.shape)
        grid_values = np.clip(grid_values, 0, None)
    
    dynamic_range = vmax / (vmin + 1e-12) if vmax > 0 else 1
    use_log = dynamic_range > 10000 and vmin > 0

    if use_log:
        levels = np.logspace(np.log10(vmin + 1e-6), np.log10(vmax), num=50)
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin + 1e-6, vmax=vmax)
        cbar_label = f'{selected_param} (Log Scale)'
    else:
        num_total = 50
        num_low = 40  # The first 40 ticks are for the low-value region (0 to vmax/2 - 1e-6)
        num_high = num_total - num_low  # The last 10 ticks are for the high-value region (vmax/2 to vmax)
        part1 = np.linspace(vmin, vmax / 2 - 1e-6, num=num_low)  # Adjust the end point of the low-value region
        part2 = np.linspace(vmax / 2, vmax, num=num_high)  # The high-value region starts normally
        levels = np.concatenate([part1, part2])
        norm = None
        cbar_label = selected_param

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_values, levels=levels, cmap='rainbow', norm=norm)
    plt.colorbar(contour, label=cbar_label, ticks=levels[::5])
    
    for region, (color, marker) in region_mapping.items():
        mask = regions == region
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                    c=color, marker=marker, s=50, edgecolor='black', 
                    linewidth=0.5, label=region)
    
    plt.title(f"Contour plot for {selected_param} (Min Value=0)")
    plt.legend(title="Regions", loc='best', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.show()

# ==== Main program call modification ====
# Define the output path (the measured_data folder in the current directory)
output_dir = Path("measured_data")
output_dir.mkdir(exist_ok=True)
excel_path = output_dir / "Measured_Point_Coordinate_Data.xlsx"

# Use ExcelWriter to write multiple worksheets at once
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Process important parameters (use the original data)
    for selected_param in top_params:
        # Key modification: Directly obtain values from the original data
        values = data1[selected_param].values
        plot_contour(tsne_result, regions, selected_param, values, region_mapping, writer)
    
    # Process other parameters (TOC and PG)
    for selected_param2 in ['TOC', 'PG']:  # Ensure the column names match
        values = data1[selected_param2].values
        plot_contour(tsne_result, regions, selected_param2, values, region_mapping, writer)

print(f"\nAll measured point XYZ data has been exported to: {excel_path.absolute()}")




# ================== 2. Parameter Importance Visualization ==================
# Calculate correlations with NMDS dimensions
param_importance = {}
for param in parameters:
    param_values = normalized_data[param].values
    corr_dim1 = np.abs(np.corrcoef(param_values, best_nmds_result[:, 0])[0, 1])
    corr_dim2 = np.abs(np.corrcoef(param_values, best_nmds_result[:, 1])[0, 1])
    param_importance[param] = (corr_dim1 + corr_dim2) / 2

# Sort and plot
# ================== Parameter Importance Visualization ==================
import seaborn as sns

# Convert data format
importance_df = pd.DataFrame(sorted_importance, columns=['Parameter', 'Importance'])

# Set plotting style
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# Plot horizontal bar chart
ax = sns.barplot(
    x='Importance',
    y='Parameter',
    data=importance_df,
    palette='viridis',  # Use a gradient color scheme
    orient='h',         # Display horizontally
    dodge=False
)

# Set labels and title
plt.title('Normalized Parameter Importance Scores', fontsize=14, pad=20)
plt.xlabel('Normalized Importance', fontsize=12)
plt.ylabel('', fontsize=12)

# Optimize axis display
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
plt.xticks(rotation=45, ha='right')  

# Add numerical labels
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 0.01,       # X position
            p.get_y() + 0.5,    # Y position
            f'{width:.2f}',     # Text content
            ha='left',          # Horizontal alignment
            va='center',        # Vertical alignment
            fontsize=9)

# Adjust layout
plt.tight_layout()
plt.show()

    # ================== 1. Dimensionality Reduction Comparison ==================
from umap import UMAP

# PCA (existing implementation)
pca_silhouette = silhouette_score(data_scaled, cluster_labels)
pca_calinski_harabasz = calinski_harabasz_score(data_scaled, cluster_labels)
pca_davies_bouldin = davies_bouldin_score(data_scaled, cluster_labels)

# UMAP implementation
umap = UMAP(n_components=2, random_state=42)
umap_result = umap.fit_transform(data_scaled)
kmeans_umap = KMeans(n_clusters=4)
umap_labels = kmeans_umap.fit_predict(umap_result)
umap_silhouette = silhouette_score(data_scaled, umap_labels)
umap_calinski = calinski_harabasz_score(data_scaled, umap_labels)
umap_davies = davies_bouldin_score(data_scaled, umap_labels)

# t-SNE (existing implementation)
tsne_silhouette, tsne_calinski_harabasz, tsne_davies_bouldin = best_scores

# Comparison table
print("\nDimensionality Reduction Comparison:")
print("Method\t\tSilhouette\tCalinski-Harabasz\tDavies-Bouldin")
print(f"PCA\t\t{pca_silhouette:.3f}\t\t{pca_calinski_harabasz:.3f}\t\t\t{pca_davies_bouldin:.3f}")
print(f"UMAP\t\t{umap_silhouette:.3f}\t\t{umap_calinski:.3f}\t\t\t{umap_davies:.3f}")
print(f"t-SNE\t\t{tsne_silhouette:.3f}\t\t{tsne_calinski_harabasz:.3f}\t\t\t{tsne_davies_bouldin:.3f}")
# ================== 1. Dimensionality Reduction Comparison ==================
from umap import UMAP

# Prepare data for plotting
methods = ['PCA', 'UMAP', 't-SNE']
scores = {
    'Silhouette': [pca_silhouette, umap_silhouette, tsne_silhouette],
    'Calinski-Harabasz': [pca_calinski_harabasz, umap_calinski, tsne_calinski_harabasz],
    'Davies-Bouldin': [pca_davies_bouldin, umap_davies, tsne_davies_bouldin]
}

# Create a three-column bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Set colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bar charts for each indicator
for ax, (score_type, values) in zip(axes, scores.items()):
    bars = ax.bar(methods, values, color=colors)
    ax.set_title(f'{score_type} Score Comparison')
    ax.set_ylabel(score_type)
    
    # Add numerical labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ================== Global Plot Styling ==================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'tiff',
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

color_palette = {
    'valid': '#2E75B6',     # Navy blue
    'invalid': '#C00000',   # Crimson red
    'DHS': '#D62728',       # Vermillion
    'GDK': '#9467BD',       # Amethyst
    'HYC': '#2CA02C',       # Green
    'JJZG': '#FF7F0E'       # Orange
}

marker_dict = {
    'DHS': 'o',  # Circle
    'GDK': 's',  # Square
    'HYC': '^',  # Triangle
    'JJZG': 'D'  # Diamond
}

# ================== 1. Fitness vs Parameters ==================
plt.figure(figsize=(6, 4.5))
ax = plt.gca()

# Add trend line
x = [h['n_valid'] for h in opt_history]
y = [h['fitness'] for h in opt_history]
z = np.polyfit(x, y, 2)
p = np.poly1d(z)
plt.plot(sorted(x), p(sorted(x)), '--', color='gray', alpha=0.7, lw=1.5, label='Trend line')

sc = ax.scatter(
    x, y,
    c=y,
    cmap='viridis',
    alpha=0.8,
    edgecolors='w',
    s=60,
    zorder=3
)

plt.colorbar(sc, label='Fitness Score', ax=ax)
plt.title('Fitness vs. Number of Valid Parameters', pad=10)
plt.xlabel('Number of Valid Parameters')
plt.ylabel('Fitness Score')
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('Fig1_Fitness_vs_Parameters.tif')
plt.show()

# ================== 2. Parameter MDS Visualization ==================
plt.figure(figsize=(7, 5))
ax = plt.gca()

# Plot valid/invalid parameters
valid_mask = np.array(param_status) == 'valid'
ax.scatter(
    param_mds_result[valid_mask, 0],
    param_mds_result[valid_mask, 1],
    c=color_palette['valid'],
    marker='o',
    s=80,
    edgecolor='w',
    label='Valid',
    zorder=3
)

ax.scatter(
    param_mds_result[~valid_mask, 0],
    param_mds_result[~valid_mask, 1],
    c=color_palette['invalid'],
    marker='X',
    s=80,
    label='Invalid',
    zorder=3
)

# Label adjustment
texts = []
for i, (x, y) in enumerate(param_mds_result):
    texts.append(ax.text(x+0.03, y+0.03, parameters[i], 
                      fontsize=8, weight='semibold', alpha=0.9))

from adjustText import adjust_text
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

plt.title('Parameter Effectiveness in MDS Space', pad=10)
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')

# --------------------------- Key adjustment: Optimize legend position---------------------------
plt.legend(
    title="Parameter Status", 
    frameon=True,          # Keep the legend border
    loc='best',            # Automatically select the position that least obscures the data
    borderpad=0.8,         # Adjust the inner padding of the legend (avoid text sticking to the edge)
    labelspacing=0.6,      # Adjust the spacing between legend items
    handlelength=1.2,      # Adjust the length of legend markers
    handletextpad=0.5,     # Adjust the spacing between markers and text
    borderaxespad=0.3,     # Adjust the spacing between the legend and the axes (ensure it is inside the figure frame)
    framealpha=0.9         # Semi-transparent legend background (reduce the sense of occlusion)
)

plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()  # Automatically optimize the overall layout (avoid the legend going out of the figure frame)
plt.savefig('Fig2_Parameter_MDS.tif')
plt.show()

# ================== 3. Distance Correlation ==================
plt.figure(figsize=(5, 4))
ax = sns.regplot(
    x=best_observed_dissimilarity.flatten(),
    y=best_nmds_result[:, 0],
    scatter_kws={'color':color_palette['valid'], 'alpha':0.6, 's':40},
    line_kws={'color':color_palette['invalid'], 'lw':1.5},
    ci=95
)

# Annotation formatting
stats_text = f'Non-metric $R^2$ = {best_non_metric_r2:.2f}\nLinear $R^2$ = {best_linear_fit_r2:.2f}'
plt.text(0.05, 0.9, stats_text, transform=ax.transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.title('Observed vs. Ordination Distances', pad=10)
plt.xlabel('Observed Dissimilarity')
plt.ylabel('NMDS Axis 1')
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('Fig3_Correlation.tif')
plt.show()

# ================== 7. Dimensionality Reduction Comparison ==================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

metric_colors = [color_palette['DHS'], color_palette['GDK'], color_palette['HYC']]

for ax, (score_type, values) in zip(axes, scores.items()):
    bars = ax.bar(methods, values, color=metric_colors, edgecolor='k')
    ax.set_title(score_type)
    ax.set_ylabel('Score Value')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height*0.95,
                f'{height:.2f}', ha='center', va='top', fontsize=8)

plt.tight_layout()
plt.savefig('Fig7_DR_Comparison.tif')
plt.show()

# ================== Export Best Hyperparameters to Excel ==================
# Collect all best hyperparameters
hyperparameters = []

#Valid parameters selected by Bayesian optimization
hyperparameters.append({
    'Method': 'Bayesian Optimization',
    'Hyperparameter': 'Selected Parameters',
    'Value': ', '.join(best_parameters)
})

# t-SNE optimized parameters
if 'best_tsne_params' in locals() or 'best_tsne_params' in globals():
    hyperparameters.append({
        'Method': 't-SNE',
        'Hyperparameter': 'Perplexity',
        'Value': best_tsne_params[0]
    })
    hyperparameters.append({
        'Method': 't-SNE',
        'Hyperparameter': 'Learning Rate',
        'Value': best_tsne_params[1]
    })
    hyperparameters.append({
        'Method': 't-SNE',
        'Hyperparameter': 'Random State',
        'Value': best_tsne_params[2]
    })
# supplement other hyperparameter records
hyperparameters.extend([
    {
        'Method': 'MDS',
        'Hyperparameter': 'n_components',
        'Value': 2
    },
    {
        'Method': 'MDS',
        'Hyperparameter': 'max_iter',
        'Value': 5000
    },
    {
        'Method': 'K-means',
        'Hyperparameter': 'n_clusters',
        'Value': 4
    },
    {
        'Method': 'Bayesian Optimization',
        'Hyperparameter': 'n_calls',
        'Value': 60
    },
    {
        'Method': 'Data Processing',
        'Hyperparameter': 'Outlier Replacement Method',
        'Value': '15th-85th percentile with linear interpolation'
    }
])

# Create DataFrame and save to Excel
hyper_df = pd.DataFrame(hyperparameters)
hyper_df.to_excel('Best_Hyperparameters for terrigenous clastic input.xlsx', index=False, columns=['Method', 'Hyperparameter', 'Value'])
print("Best hyperparameters have been saved to Best_Hyperparameters for terrigenous clastic input.xlsx")