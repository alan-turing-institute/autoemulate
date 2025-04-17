import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

def plot_implausibility_distribution(impl_scores, threshold=3.0, figsize=(10, 6)):
    """
    Plot the distribution of implausibility scores
    
    Args:
        impl_scores: Array of implausibility scores from history matching
        threshold: Implausibility threshold used for NROY points
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Flatten array if it's 2D
    if len(impl_scores.shape) > 1:
        flat_scores = impl_scores.flatten()
    else:
        flat_scores = impl_scores
    
    # Create histogram
    sns.histplot(flat_scores, kde=True)
    
    # Add vertical line for threshold
    plt.axvline(x=threshold, color='r', linestyle='--', 
                label=f'Threshold = {threshold}')
    
    # Calculate percentage below threshold
    below_threshold = (flat_scores <= threshold).sum() / len(flat_scores) * 100
    
    plt.title(f'Implausibility Score Distribution\n'
              f'{below_threshold:.1f}% of points below threshold')
    plt.xlabel('Implausibility Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def plot_parameter_vs_implausibility(samples, impl_scores, param_names, threshold=3.0, 
                                    nrows=None, ncols=None, figsize=(15, 10)):
    """
    Plot each parameter against minimum implausibility score
    
    Args:
        samples: List of parameter dictionaries or dataframe
        impl_scores: Array of implausibility scores
        param_names: List of parameter names to plot
        threshold: Implausibility threshold
        nrows, ncols: Plot grid dimensions (calculated automatically if None)
        figsize: Figure size
    """
    # Convert samples to dataframe if it's a list
    if isinstance(samples, list):
        df = pd.DataFrame(samples)
    else:
        df = samples.copy()
    
    # Calculate minimum implausibility for each sample
    if len(impl_scores.shape) > 1:
        min_impl = np.min(impl_scores, axis=1)
    else:
        min_impl = impl_scores
    
    df['min_implausibility'] = min_impl
    df['NROY'] = min_impl <= threshold
    
    # Determine grid size
    if nrows is None or ncols is None:
        n_params = len(param_names)
        ncols = min(3, n_params)
        nrows = int(np.ceil(n_params / ncols))
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each parameter
    for i, param in enumerate(param_names):
        if i < len(axes):
            ax = axes[i]
            
            # Create scatter plot
            sc = ax.scatter(df[param], df['min_implausibility'], 
                          c=df['NROY'], cmap='viridis', 
                          alpha=0.7, s=30)
            
            # Add horizontal line for threshold
            ax.axhline(y=threshold, color='r', linestyle='--',
                      label=f'Threshold = {threshold}')
            
            ax.set_title(f'Parameter: {param}')
            ax.set_xlabel(param)
            ax.set_ylabel('Min Implausibility')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Ruled Out', 'NROY'])
    
    plt.suptitle('Parameters vs. Minimum Implausibility', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    return plt

def plot_pairwise_parameters(samples, impl_scores, param_names=None, threshold=3.0,
                           max_pairs=6, figsize=(15, 12)):
    """
    Plot pairwise parameter combinations colored by implausibility
    
    Args:
        samples: List of parameter dictionaries or dataframe
        impl_scores: Array of implausibility scores
        param_names: List of parameter names to plot (uses all if None)
        threshold: Implausibility threshold
        max_pairs: Maximum number of parameter pairs to plot
        figsize: Figure size
    """
    # Convert samples to dataframe if it's a list
    if isinstance(samples, list):
        df = pd.DataFrame(samples)
    else:
        df = samples.copy()
    
    # Use all parameters if param_names is None
    if param_names is None:
        # Exclude potential non-parameter columns
        exclude_cols = ['min_implausibility', 'NROY']
        param_names = [col for col in df.columns if col not in exclude_cols]
    
    # Calculate minimum implausibility for each sample
    if len(impl_scores.shape) > 1:
        min_impl = np.min(impl_scores, axis=1)
    else:
        min_impl = impl_scores
    
    df['min_implausibility'] = min_impl
    df['NROY'] = min_impl <= threshold
    
    # Determine number of pairs to plot
    n_params = len(param_names)
    n_pairs = min(max_pairs, n_params * (n_params - 1) // 2)
    
    # Create custom colormap: green for NROY, grey for ruled out
    colors = [(0.7, 0.7, 0.7, 0.7), (0.0, 0.8, 0.0, 1.0)]
    cmap = LinearSegmentedColormap.from_list('nroy_cmap', colors, N=2)
    
    # Determine grid size
    ncols = min(3, n_pairs)
    nrows = int(np.ceil(n_pairs / ncols))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Counter for subplot position
    pair_count = 0
    
    # Create pairwise plots
    for i in range(n_params):
        for j in range(i+1, n_params):
            if pair_count < n_pairs:
                # Create subplot
                ax = fig.add_subplot(nrows, ncols, pair_count + 1)
                
                # Create scatter plot
                sc = ax.scatter(df[param_names[i]], df[param_names[j]],
                              c=df['NROY'], cmap=cmap,
                              alpha=0.7, s=30)
                
                # Add labels
                ax.set_xlabel(param_names[i])
                ax.set_ylabel(param_names[j])
                ax.grid(True, alpha=0.3)
                
                # Highlight NROY region boundaries
                if df['NROY'].sum() > 0:
                    nroy_df = df[df['NROY']]
                    ax.axvline(x=nroy_df[param_names[i]].min(), color='g', linestyle='--', alpha=0.5)
                    ax.axvline(x=nroy_df[param_names[i]].max(), color='g', linestyle='--', alpha=0.5)
                    ax.axhline(y=nroy_df[param_names[j]].min(), color='g', linestyle='--', alpha=0.5)
                    ax.axhline(y=nroy_df[param_names[j]].max(), color='g', linestyle='--', alpha=0.5)
                
                pair_count += 1
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Ruled Out', 'NROY'])
    
    plt.suptitle('Pairwise Parameter Combinations', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    return plt

def plot_wave_convergence(wave_results, param_names=None, figsize=(14, 10)):
    """
    Plot parameter range reduction across waves
    
    Args:
        wave_results: List of (samples, impl_scores) tuples for each wave
        param_names: List of parameter names to plot (uses all if None)
        figsize: Figure size
    """
    # Extract samples from each wave
    all_wave_samples = [samples for samples, _ in wave_results]
    
    # Convert first wave samples to dataframe to get parameter names
    if len(all_wave_samples) > 0:
        if isinstance(all_wave_samples[0], list) and len(all_wave_samples[0]) > 0 and isinstance(all_wave_samples[0][0], dict):
            wave0_df = pd.DataFrame(all_wave_samples[0])
        else:
            # Assume it's already a DataFrame
            wave0_df = all_wave_samples[0].copy() if hasattr(all_wave_samples[0], 'copy') else pd.DataFrame(all_wave_samples[0])
    else:
        # No waves to plot
        plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, 'No wave data available to plot', ha='center', va='center', fontsize=14)
        plt.axis('off')
        return plt
    
    # Use all parameters if param_names is None
    if param_names is None:
        # Exclude potential non-parameter columns
        exclude_cols = ['min_implausibility', 'NROY', 'max_implausibility']
        param_names = [col for col in wave0_df.columns if col not in exclude_cols]
    
    # Number of waves
    n_waves = len(all_wave_samples)
    
    # Create figure
    fig, axes = plt.subplots(len(param_names), 1, figsize=figsize, sharex=True)
    if len(param_names) == 1:
        axes = [axes]
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        # Track min/max ranges for each wave
        wave_mins, wave_maxs, wave_medians = [], [], []
        
        for wave_idx, wave_samples in enumerate(all_wave_samples):
            # Convert to dataframe if needed
            if isinstance(wave_samples, list) and len(wave_samples) > 0 and isinstance(wave_samples[0], dict):
                wave_df = pd.DataFrame(wave_samples)
            else:
                # Assume it's already a DataFrame
                wave_df = wave_samples.copy() if hasattr(wave_samples, 'copy') else pd.DataFrame(wave_samples)
            
            # Get parameter stats if the parameter exists in this wave
            if param in wave_df.columns:
                param_min = wave_df[param].min()
                param_max = wave_df[param].max()
                param_median = wave_df[param].median()
            else:
                # Parameter not in this wave, use values from previous wave or default
                param_min = wave_mins[-1] if wave_mins else 0
                param_max = wave_maxs[-1] if wave_maxs else 1
                param_median = wave_medians[-1] if wave_medians else 0.5
            
            wave_mins.append(param_min)
            wave_maxs.append(param_max)
            wave_medians.append(param_median)
        
        # Plot parameter ranges
        x = np.arange(n_waves)
        ax.fill_between(x, wave_mins, wave_maxs, alpha=0.3, color='blue')
        ax.plot(x, wave_medians, 'o-', color='blue', label='Median')
        
        # Calculate normalized range reduction
        initial_range = wave_maxs[0] - wave_mins[0]
        final_range = wave_maxs[-1] - wave_mins[-1]
        reduction = (1 - final_range / initial_range) * 100 if initial_range > 0 else 0
        
        ax.set_ylabel(param)
        ax.set_title(f'{param}: {reduction:.1f}% range reduction')
        ax.grid(True, alpha=0.3)
    
    # Set x-axis labels
    axes[-1].set_xlabel('Wave')
    axes[-1].set_xticks(np.arange(n_waves))
    axes[-1].set_xticklabels([f'Wave {i+1}' for i in range(n_waves)])
    
    plt.suptitle('Parameter Range Reduction Across Waves', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return plt

def plot_3d_nroy(samples, impl_scores, param_names, threshold=3.0, figsize=(12, 10)):
    """
    Create a 3D visualization of the NROY space for three parameters
    
    Args:
        samples: List of parameter dictionaries or dataframe
        impl_scores: Array of implausibility scores
        param_names: List of exactly 3 parameter names to plot
        threshold: Implausibility threshold
        figsize: Figure size
    """
    if len(param_names) != 3:
        raise ValueError("Exactly 3 parameter names must be provided for 3D plot")
    
    # Convert samples to dataframe if it's a list
    if isinstance(samples, list):
        df = pd.DataFrame(samples)
    else:
        df = samples.copy()
    
    # Calculate minimum implausibility for each sample
    if len(impl_scores.shape) > 1:
        min_impl = np.min(impl_scores, axis=1)
    else:
        min_impl = impl_scores
    
    df['min_implausibility'] = min_impl
    df['NROY'] = min_impl <= threshold
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract parameters
    x = df[param_names[0]]
    y = df[param_names[1]]
    z = df[param_names[2]]
    
    # Create scatter plot
    scatter = ax.scatter(
        x, y, z,
        c=df['min_implausibility'],
        cmap='viridis_r',  # Reversed viridis so green is low implausibility
        s=30,
        alpha=0.7
    )
    
    # Highlight NROY points
    if df['NROY'].sum() > 0:
        nroy_df = df[df['NROY']]
        ax.scatter(
            nroy_df[param_names[0]],
            nroy_df[param_names[1]],
            nroy_df[param_names[2]],
            color='green',
            s=50,
            alpha=1.0,
            marker='o',
            label='NROY Points'
        )
    
    # Add labels
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_zlabel(param_names[2])
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Implausibility')
    
    # Add threshold line on colorbar
    cbar.ax.axhline(y=threshold, color='r', linestyle='--')
    
    plt.title('3D Visualization of NROY Space', fontsize=16)
    plt.tight_layout()
    
    return plt

def plot_pca_nroy(samples, impl_scores, threshold=3.0, n_components=2, figsize=(12, 10)):
    """
    Plot NROY space using PCA for dimensionality reduction
    
    Args:
        samples: List of parameter dictionaries or dataframe
        impl_scores: Array of implausibility scores
        threshold: Implausibility threshold
        n_components: Number of PCA components to use (2 or 3)
        figsize: Figure size
    """
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3")
    
    # Convert samples to dataframe if it's a list
    if isinstance(samples, list):
        df = pd.DataFrame(samples)
    else:
        df = samples.copy()
    
    # Remove any non-parameter columns
    exclude_cols = ['min_implausibility', 'NROY']
    param_df = df[[col for col in df.columns if col not in exclude_cols]]
    
    # Calculate minimum implausibility for each sample
    if len(impl_scores.shape) > 1:
        min_impl = np.min(impl_scores, axis=1)
    else:
        min_impl = impl_scores
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(param_df)
    
    # Create dataframe with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['min_implausibility'] = min_impl
    pca_df['NROY'] = min_impl <= threshold
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    if n_components == 2:
        # 2D PCA plot
        ax = fig.add_subplot(111)
        
        # Create scatter plot
        scatter = ax.scatter(
            pca_df['PC1'], 
            pca_df['PC2'],
            c=pca_df['min_implausibility'],
            cmap='viridis_r',
            s=30,
            alpha=0.7
        )
        
        # Highlight NROY points
        if pca_df['NROY'].sum() > 0:
            nroy_df = pca_df[pca_df['NROY']]
            ax.scatter(
                nroy_df['PC1'],
                nroy_df['PC2'],
                color='green',
                s=50,
                alpha=1.0,
                marker='o',
                label='NROY Points'
            )
        
        # Add labels
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1f}% explained variance)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1f}% explained variance)')
        
    else:
        # 3D PCA plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        scatter = ax.scatter(
            pca_df['PC1'], 
            pca_df['PC2'],
            pca_df['PC3'],
            c=pca_df['min_implausibility'],
            cmap='viridis_r',
            s=30,
            alpha=0.7
        )
        
        # Highlight NROY points
        if pca_df['NROY'].sum() > 0:
            nroy_df = pca_df[pca_df['NROY']]
            ax.scatter(
                nroy_df['PC1'],
                nroy_df['PC2'],
                nroy_df['PC3'],
                color='green',
                s=50,
                alpha=1.0,
                marker='o',
                label='NROY Points'
            )
        
        # Add labels
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1f}% explained variance)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1f}% explained variance)')
        ax.set_zlabel(f'PC3 ({explained_var[2]:.1f}% explained variance)')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Implausibility')
    
    # Add threshold line on colorbar
    cbar.ax.axhline(y=threshold, color='r', linestyle='--')
    
    plt.title(f'PCA Visualization of NROY Space ({n_components}D)', fontsize=16)
    plt.tight_layout()
    
    # Add parameter loading information in a separate plot
    if hasattr(pca, 'components_'):
        param_names = param_df.columns
        
        # Create a new figure for loadings
        fig_loadings, ax_loadings = plt.subplots(figsize=(12, 6))
        
        # Display loadings as a heatmap
        loadings = pca.components_.T
        sns.heatmap(loadings, annot=True, cmap='coolwarm', 
                  xticklabels=[f'PC{i+1}' for i in range(n_components)],
                  yticklabels=param_names, ax=ax_loadings)
        
        ax_loadings.set_title('PCA Component Loadings')
        plt.tight_layout()
    
    return plt

def plot_implausibility_radar(samples, impl_scores, output_names, sample_indices=None, 
                            threshold=3.0, figsize=(12, 10)):
    """
    Create radar plots showing implausibility for different outputs
    
    Args:
        samples: List of parameter dictionaries or dataframe
        impl_scores: Array of implausibility scores with shape (n_samples, n_outputs)
        output_names: List of output names
        sample_indices: Indices of samples to plot (plots best samples if None)
        threshold: Implausibility threshold
        figsize: Figure size
    """
    # Make sure impl_scores is 2D
    if len(impl_scores.shape) == 1:
        impl_scores = impl_scores.reshape(-1, 1)
    
    n_samples, n_outputs = impl_scores.shape
    
    if len(output_names) != n_outputs:
        raise ValueError(f"Number of output names ({len(output_names)}) must match "
                        f"number of outputs in impl_scores ({n_outputs})")
    
    # If no sample indices provided, find samples with lowest max implausibility
    if sample_indices is None:
        max_impl = np.max(impl_scores, axis=1)
        # Get indices of 5 best samples (lowest max implausibility)
        sample_indices = np.argsort(max_impl)[:5]
    
    n_plots = len(sample_indices)
    
    # Set up angles for radar plot
    angles = np.linspace(0, 2*np.pi, n_outputs, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create extended output names and scores arrays (for closing the loop)
    extended_output_names = output_names + [output_names[0]]
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, subplot_kw=dict(polar=True), figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Plot each sample
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Get scores for this sample
        scores = impl_scores[idx].tolist()
        scores += scores[:1]  # Close the loop
        
        # Plot threshold
        ax.plot(angles, [threshold]*len(angles), 'r--', linewidth=1, label='Threshold')
        
        # Fill the area below threshold
        ax.fill(angles, [threshold]*len(angles), 'r', alpha=0.1)
        
        # Plot implausibility
        ax.plot(angles, scores, 'b-', linewidth=2, label='Implausibility')
        ax.fill(angles, scores, 'b', alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(output_names, fontsize=8)
        
        # Set y limits
        max_score = max(max(scores), threshold * 1.5)
        ax.set_ylim(0, max_score)
        
        # Get parameter values for the sample
        if isinstance(samples, list):
            sample_params = samples[idx]
        else:
            sample_params = samples.iloc[idx].to_dict()
        
        # Remove non-parameter entries if present
        for key in ['min_implausibility', 'NROY']:
            if key in sample_params:
                del sample_params[key]
        
        # Create title with parameter values
        param_strs = [f"{key}={value:.3g}" for key, value in sample_params.items()]
        param_title = '\n'.join([', '.join(param_strs[i:i+3]) for i in range(0, len(param_strs), 3)])
        
        ax.set_title(f"Sample {idx}\n{param_title}", fontsize=9)
    
    plt.suptitle('Implausibility Radar Plots for Selected Samples', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return plt

def visualize_history_matching_results(samples_list, impl_scores_list, param_names, output_names, 
                                     threshold=3.0, figsize_base=(12, 10), max_waves=3):
    """
    Create a comprehensive visualization of history matching results
    
    Args:
        samples_list: List of samples for each wave
        impl_scores_list: List of implausibility scores for each wave
        param_names: List of parameter names
        output_names: List of output names
        threshold: Implausibility threshold
        figsize_base: Base figure size to scale from
        max_waves: Maximum number of waves to show detailed plots for
    """
    n_waves = len(samples_list)
    
    if n_waves == 0:
        print("No waves to visualize.")
        return "No visualization created"
    
    # Check if param_names and output_names are valid
    if not param_names or len(param_names) == 0:
        print("Warning: No parameter names provided.")
        # Try to extract from first sample
        if hasattr(samples_list[0], 'columns'):
            param_names = list(samples_list[0].columns)
        else:
            print("Cannot create visualizations without parameter names.")
            return "Visualization failed"
    
    if not output_names or len(output_names) == 0:
        print("Warning: No output names provided.")
        # We can still create some visualizations without output names
    
    try:
        # Create a multi-page set of visualizations
        
        # 1. Wave convergence plot (all waves)
        wave_results = list(zip(samples_list, impl_scores_list))
        plt_wave = plot_wave_convergence(wave_results, param_names, figsize=(figsize_base[0], figsize_base[1]*0.8))
        plt_wave.savefig('wave_convergence.png', dpi=300, bbox_inches='tight')
        plt_wave.close()
        
        # 2. Parameter vs implausibility plots (final wave)
        plt_param_impl = plot_parameter_vs_implausibility(
            samples_list[-1], impl_scores_list[-1], param_names, threshold,
            figsize=(figsize_base[0], figsize_base[1]*0.8)
        )
        plt_param_impl.savefig('param_vs_implausibility.png', dpi=300, bbox_inches='tight')
        plt_param_impl.close()
        
        # 3. Pairwise parameter plots (final wave)
        if len(param_names) >= 2:
            plt_pairwise = plot_pairwise_parameters(
                samples_list[-1], impl_scores_list[-1], param_names, threshold,
                figsize=(figsize_base[0], figsize_base[1])
            )
            plt_pairwise.savefig('pairwise_parameters.png', dpi=300, bbox_inches='tight')
            plt_pairwise.close()
        
        # 4. Implausibility distribution for each wave
        for i, impl_scores in enumerate(impl_scores_list[:min(max_waves, n_waves)]):
            plt_dist = plot_implausibility_distribution(
                impl_scores, threshold, figsize=(figsize_base[0]*0.8, figsize_base[1]*0.5)
            )
            plt_dist.savefig(f'implausibility_dist_wave{i+1}.png', dpi=300, bbox_inches='tight')
            plt_dist.close()
        
        # 5. PCA visualization (final wave)
        if len(param_names) > 2:
            try:
                plt_pca = plot_pca_nroy(
                    samples_list[-1], impl_scores_list[-1], threshold, 
                    n_components=min(3, len(param_names)),
                    figsize=(figsize_base[0], figsize_base[1])
                )
                plt_pca.savefig('pca_nroy.png', dpi=300, bbox_inches='tight')
                plt_pca.close()
            except Exception as e:
                print(f"Error creating PCA visualization: {e}")
        
        # 6. Radar plots for best samples (final wave)
        if output_names and len(output_names) > 0:
            try:
                plt_radar = plot_implausibility_radar(
                    samples_list[-1], impl_scores_list[-1], output_names, 
                    figsize=(figsize_base[0]*1.2, figsize_base[1]*0.5)
                )
                plt_radar.savefig('implausibility_radar.png', dpi=300, bbox_inches='tight')
                plt_radar.close()
            except Exception as e:
                print(f"Error creating radar plot: {e}")
        
        # 7. 3D visualization for selected parameters (if at least 3 parameters)
        if len(param_names) >= 3:
            # Choose 3 most important parameters (could be improved with sensitivity analysis)
            selected_params = param_names[:3]
            try:
                plt_3d = plot_3d_nroy(
                    samples_list[-1], impl_scores_list[-1], selected_params, threshold,
                    figsize=(figsize_base[0], figsize_base[1])
                )
                plt_3d.savefig('nroy_3d.png', dpi=300, bbox_inches='tight')
                plt_3d.close()
            except Exception as e:
                print(f"Error creating 3D visualization: {e}")
        
        print("Visualizations saved to disk:")
        print("1. wave_convergence.png - Parameter range reduction across waves")
        print("2. param_vs_implausibility.png - Parameters vs implausibility scores")
        if len(param_names) >= 2:
            print("3. pairwise_parameters.png - Pairwise parameter visualizations")
        print(f"4. implausibility_dist_wave1.png to implausibility_dist_wave{min(n_waves, max_waves)}.png - Implausibility distributions")
        if len(param_names) > 2:
            print("5. pca_nroy.png - PCA visualization of NROY space")
        if output_names and len(output_names) > 0:
            print("6. implausibility_radar.png - Radar plots of implausibility for best samples")
        if len(param_names) >= 3:
            print("7. nroy_3d.png - 3D visualization of NROY space")
        
        return "Visualization complete"
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return "Visualization failed"

# Helper function for using with HistoryMatcher class
def plot_history_matching_results(hm, all_samples, all_impl_scores, n_waves=3):
    """
    Plot history matching results using the HistoryMatcher object
    
    Args:
        hm: HistoryMatcher instance
        all_samples: List of parameter samples (output from run_history_matching)
        all_impl_scores: Array of implausibility scores (output from run_history_matching)
        n_waves: Number of waves in the history matching run
    """
    # Convert all_samples to DataFrame if it's a list of dictionaries
    if isinstance(all_samples, list) and len(all_samples) > 0 and isinstance(all_samples[0], dict):
        samples_df = pd.DataFrame(all_samples)
    else:
        # Assume it's already a DataFrame
        samples_df = all_samples.copy() if hasattr(all_samples, 'copy') else pd.DataFrame(all_samples)
    
    # Get parameter and output names from HistoryMatcher
    param_names = hm.simulator.param_names
    output_names = hm.simulator.output_names
    
    # Check if we have samples to plot
    if len(samples_df) == 0:
        print("No samples to plot.")
        return [], []
    
    # Split the samples and scores into waves
    samples_per_wave = max(1, len(samples_df) // n_waves)
    samples_list = []
    impl_scores_list = []
    
    for i in range(n_waves):
        start_idx = i * samples_per_wave
        end_idx = (i + 1) * samples_per_wave if i < n_waves - 1 else len(samples_df)
        
        # Handle case where there are fewer samples than expected
        if start_idx >= len(samples_df):
            break
            
        wave_samples = samples_df.iloc[start_idx:end_idx]
        
        # Handle case where all_impl_scores is not numpy array
        if isinstance(all_impl_scores, list):
            all_impl_scores = np.array(all_impl_scores)
            
        # Make sure we don't go out of bounds
        end_idx_scores = min(end_idx, len(all_impl_scores))
        wave_scores = all_impl_scores[start_idx:end_idx_scores]
        
        samples_list.append(wave_samples)
        impl_scores_list.append(wave_scores)
    
    # Create visualizations only if we have waves to plot
    if len(samples_list) > 0:
        visualize_history_matching_results(
            samples_list, 
            impl_scores_list, 
            param_names, 
            output_names, 
            threshold=hm.threshold
        )
    else:
        print("No complete waves to visualize.")
    
    return samples_list, impl_scores_list