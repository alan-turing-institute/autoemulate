import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA


class HistoryMatchingDashboard:
    """
    Interactive dashboard for exploring history matching results
    """

    def __init__(self, samples, impl_scores, param_names, output_names, threshold=3.0):
        """
        Initialize the dashboard

        Args:
            samples: DataFrame or list of dictionaries with parameter samples
            impl_scores: Array of implausibility scores
            param_names: List of parameter names
            output_names: List of output names
            threshold: Implausibility threshold
        """
        # Convert samples to DataFrame if it's a list
        if isinstance(samples, list):
            self.samples_df = pd.DataFrame(samples)
        else:
            self.samples_df = samples.copy()

        # Store other data
        self.impl_scores = impl_scores
        self.param_names = param_names
        self.output_names = output_names
        self.threshold = threshold

        # Calculate minimum implausibility for each sample
        if len(impl_scores.shape) > 1:
            self.min_impl = np.min(impl_scores, axis=1)
            self.max_impl = np.max(impl_scores, axis=1)
        else:
            self.min_impl = impl_scores
            self.max_impl = impl_scores

        # Add implausibility to DataFrame
        self.samples_df["min_implausibility"] = self.min_impl
        self.samples_df["max_implausibility"] = self.max_impl
        self.samples_df["NROY"] = self.max_impl <= threshold

        # Create the UI elements
        self._create_ui()

    def _create_ui(self):
        """Create the user interface widgets"""
        # Plot type selection
        self.plot_type = widgets.Dropdown(
            options=[
                "Parameter vs Implausibility",
                "Pairwise Parameters",
                "Implausibility Distribution",
                "NROY Parameter Ranges",
                "Parameter Correlation Heatmap",
                "PCA Visualization",
                "3D Parameter Visualization",
                "Implausibility Radar",
            ],
            value="Parameter vs Implausibility",
            description="Plot Type:",
            style={"description_width": "initial"},
        )

        # Parameter selection widgets
        self.param_x = widgets.Dropdown(
            options=self.param_names,
            value=self.param_names[0] if self.param_names else None,
            description="X Parameter:",
            disabled=False,
        )

        self.param_y = widgets.Dropdown(
            options=self.param_names,
            value=self.param_names[1]
            if len(self.param_names) > 1
            else self.param_names[0],
            description="Y Parameter:",
            disabled=False,
        )

        self.param_z = widgets.Dropdown(
            options=self.param_names,
            value=self.param_names[2]
            if len(self.param_names) > 2
            else self.param_names[0],
            description="Z Parameter:",
            disabled=False,
        )

        # Threshold slider
        self.threshold_slider = widgets.FloatSlider(
            value=self.threshold,
            min=0.5,
            max=10.0,
            step=0.1,
            description="Threshold:",
            continuous_update=False,
        )

        # Sample selection for radar plot
        self.sample_selector = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.samples_df) - 1,
            step=1,
            description="Sample Index:",
            continuous_update=False,
        )

        # Wave filter (assuming waves are chronological in the data)
        n_samples = len(self.samples_df)
        estimated_waves = min(5, n_samples // 20) if n_samples > 20 else 1

        self.wave_selector = widgets.IntRangeSlider(
            value=(1, estimated_waves),
            min=1,
            max=estimated_waves,
            step=1,
            description="Waves:",
            continuous_update=False,
            readout=True,
        )

        # NROY filter
        self.nroy_filter = widgets.Checkbox(
            value=False, description="Show only NROY points", disabled=False
        )

        # Update button
        self.update_button = widgets.Button(
            description="Update Plot",
            button_style="primary",
            tooltip="Click to update the plot",
        )
        self.update_button.on_click(self._update_plot)

        # Output area for the plot
        self.output = widgets.Output()

        # Layout the UI
        param_selectors = widgets.HBox([self.param_x, self.param_y, self.param_z])
        controls_top = widgets.HBox([self.plot_type, self.threshold_slider])
        controls_bottom = widgets.HBox(
            [self.sample_selector, self.wave_selector, self.nroy_filter]
        )

        # Main layout
        self.main_layout = widgets.VBox(
            [
                controls_top,
                param_selectors,
                controls_bottom,
                self.update_button,
                self.output,
            ]
        )

    def _update_plot(self, _):
        """Update the plot based on current widget values"""
        with self.output:
            clear_output(wait=True)

            # Filter data based on wave selection
            wave_start = self.wave_selector.value[0]
            wave_end = self.wave_selector.value[1]

            n_samples = len(self.samples_df)
            samples_per_wave = n_samples // self.wave_selector.max

            start_idx = (wave_start - 1) * samples_per_wave
            end_idx = min(wave_end * samples_per_wave, n_samples)

            filtered_df = self.samples_df.iloc[start_idx:end_idx].copy()
            filtered_scores = self.impl_scores[start_idx:end_idx].copy()

            # Apply NROY filter if selected
            if self.nroy_filter.value:
                nroy_mask = filtered_df["NROY"] == True
                filtered_df = filtered_df[nroy_mask].copy()
                filtered_scores = filtered_scores[nroy_mask.values].copy()

            # Update threshold
            threshold = self.threshold_slider.value
            filtered_df["NROY"] = (
                np.max(filtered_scores, axis=1) <= threshold
                if len(filtered_scores.shape) > 1
                else filtered_scores <= threshold
            )

            # Check if we have data to plot
            if len(filtered_df) == 0:
                plt.figure(figsize=(10, 6))
                plt.text(
                    0.5,
                    0.5,
                    "No data to display with current filters",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                plt.axis("off")
                plt.tight_layout()
                plt.show()
                return

            # Generate the selected plot
            plot_type = self.plot_type.value

            try:
                if plot_type == "Parameter vs Implausibility":
                    self._plot_parameter_vs_implausibility(filtered_df, filtered_scores)
                elif plot_type == "Pairwise Parameters":
                    self._plot_pairwise_parameters(filtered_df, filtered_scores)
                elif plot_type == "Implausibility Distribution":
                    self._plot_implausibility_distribution(filtered_scores)
                elif plot_type == "NROY Parameter Ranges":
                    self._plot_nroy_parameter_ranges(filtered_df)
                elif plot_type == "Parameter Correlation Heatmap":
                    self._plot_parameter_correlation(filtered_df)
                elif plot_type == "PCA Visualization":
                    self._plot_pca_visualization(filtered_df, filtered_scores)
                elif plot_type == "3D Parameter Visualization":
                    self._plot_3d_visualization(filtered_df, filtered_scores)
                elif plot_type == "Implausibility Radar":
                    self._plot_implausibility_radar(filtered_df, filtered_scores)
            except Exception as e:
                plt.figure(figsize=(10, 6))
                plt.text(
                    0.5,
                    0.5,
                    f"Error generating plot: {str(e)}",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                plt.axis("off")
                plt.tight_layout()
                plt.show()

    def _plot_parameter_vs_implausibility(self, df, impl_scores):
        """Plot parameter vs implausibility"""
        threshold = self.threshold_slider.value
        param = self.param_x.value

        plt.figure(figsize=(12, 6))

        # Calculate minimum implausibility if we have multiple outputs
        if len(impl_scores.shape) > 1:
            min_impl = np.min(impl_scores, axis=1)
            max_impl = np.max(impl_scores, axis=1)
        else:
            min_impl = impl_scores
            max_impl = impl_scores

        # Create scatter plot
        plt.scatter(
            df[param],
            min_impl,
            c=max_impl,
            cmap="viridis_r",
            alpha=0.7,
            s=50,
            label="Min Implausibility",
        )

        # Add horizontal line for threshold
        plt.axhline(
            y=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}"
        )

        plt.title(f"Parameter {param} vs Implausibility")
        plt.xlabel(param)
        plt.ylabel("Implausibility")
        plt.colorbar(label="Max Implausibility")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_pairwise_parameters(self, df, impl_scores):
        """Plot pairwise parameter visualization"""
        threshold = self.threshold_slider.value
        param_x = self.param_x.value
        param_y = self.param_y.value

        plt.figure(figsize=(10, 8))

        # Calculate maximum implausibility
        if len(impl_scores.shape) > 1:
            max_impl = np.max(impl_scores, axis=1)
        else:
            max_impl = impl_scores

        # Create scatter plot
        sc = plt.scatter(
            df[param_x], df[param_y], c=max_impl, cmap="viridis_r", alpha=0.7, s=50
        )

        # Highlight NROY points with an outline
        nroy_points = df[df["NROY"] == True]
        if not nroy_points.empty:
            plt.scatter(
                nroy_points[param_x],
                nroy_points[param_y],
                s=80,
                facecolors="none",
                edgecolors="g",
                linewidths=2,
                label="NROY Points",
            )

            # Add NROY region boundaries
            plt.axvline(
                x=nroy_points[param_x].min(), color="g", linestyle="--", alpha=0.5
            )
            plt.axvline(
                x=nroy_points[param_x].max(), color="g", linestyle="--", alpha=0.5
            )
            plt.axhline(
                y=nroy_points[param_y].min(), color="g", linestyle="--", alpha=0.5
            )
            plt.axhline(
                y=nroy_points[param_y].max(), color="g", linestyle="--", alpha=0.5
            )

        plt.title(f"Parameters {param_x} vs {param_y}")
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.colorbar(sc, label="Max Implausibility")
        plt.grid(True, alpha=0.3)
        if not nroy_points.empty:
            plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_implausibility_distribution(self, impl_scores):
        """Plot implausibility distribution"""
        threshold = self.threshold_slider.value

        plt.figure(figsize=(12, 6))

        # Flatten array if it's 2D (multiple outputs)
        if len(impl_scores.shape) > 1:
            # Plot distribution of maximum implausibility
            max_impl = np.max(impl_scores, axis=1)
            min_impl = np.min(impl_scores, axis=1)

            # Create histograms
            plt.hist(max_impl, bins=30, alpha=0.7, label="Max Implausibility")
            plt.hist(min_impl, bins=30, alpha=0.5, label="Min Implausibility")
        else:
            # Single output case
            plt.hist(impl_scores, bins=30, alpha=0.7, label="Implausibility")

        # Add vertical line for threshold
        plt.axvline(
            x=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}"
        )

        # Calculate percentage below threshold
        if len(impl_scores.shape) > 1:
            below_threshold = (max_impl <= threshold).sum() / len(max_impl) * 100
        else:
            below_threshold = (impl_scores <= threshold).sum() / len(impl_scores) * 100

        plt.title(
            f"Implausibility Distribution\n"
            f"{below_threshold:.1f}% of points below threshold"
        )
        plt.xlabel("Implausibility")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_nroy_parameter_ranges(self, df):
        """Plot NROY parameter ranges"""
        # Filter for NROY points
        nroy_df = df[df["NROY"] == True]

        if nroy_df.empty:
            plt.figure(figsize=(12, 6))
            plt.text(
                0.5,
                0.5,
                "No NROY points found with current threshold",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            return

        # Get parameter ranges
        param_names = [p for p in self.param_names if p in df.columns]
        n_params = len(param_names)

        # Create violin plot
        plt.figure(figsize=(12, 8))

        # Create subplots for each parameter
        for i, param in enumerate(param_names):
            plt.subplot(n_params, 1, i + 1)

            # Get min/max values for the parameter
            original_min = df[param].min()
            original_max = df[param].max()
            nroy_min = nroy_df[param].min()
            nroy_max = nroy_df[param].max()

            # Calculate range reduction
            original_range = original_max - original_min
            nroy_range = nroy_max - nroy_min
            reduction = (
                (1 - nroy_range / original_range) * 100 if original_range > 0 else 0
            )

            # Create range plot
            plt.plot(
                [original_min, original_max],
                [0, 0],
                "b-",
                linewidth=10,
                alpha=0.3,
                label="Original Range",
            )
            plt.plot(
                [nroy_min, nroy_max],
                [0, 0],
                "g-",
                linewidth=10,
                alpha=0.7,
                label="NROY Range",
            )

            # Add points for all NROY values
            plt.plot(
                nroy_df[param],
                np.zeros_like(nroy_df[param]),
                "g|",
                markersize=15,
                alpha=0.5,
            )

            plt.title(f"{param}: {reduction:.1f}% range reduction")
            plt.yticks([])

            # Only show legend on first subplot
            if i == 0:
                plt.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def _plot_parameter_correlation(self, df):
        """Plot parameter correlation heatmap"""
        # Get only parameter columns
        param_names = [p for p in self.param_names if p in df.columns]
        params_df = df[param_names]

        # Calculate correlation matrix
        corr = params_df.corr()

        # Create heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        # Add correlation values
        for i in range(len(corr)):
            for j in range(len(corr)):
                plt.text(
                    j,
                    i,
                    f"{corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                )

        # Add labels
        plt.xticks(np.arange(len(param_names)), param_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(param_names)), param_names)

        plt.title("Parameter Correlation Heatmap")
        plt.colorbar(label="Correlation")
        plt.tight_layout()
        plt.show()

    def _plot_pca_visualization(self, df, impl_scores):
        """Plot PCA visualization of parameter space"""
        threshold = self.threshold_slider.value

        # Get only parameter columns
        param_names = [p for p in self.param_names if p in df.columns]
        params_df = df[param_names]

        # Calculate max implausibility
        if len(impl_scores.shape) > 1:
            max_impl = np.max(impl_scores, axis=1)
        else:
            max_impl = impl_scores

        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(params_df)

        # Create dataframe with PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
        pca_df["max_implausibility"] = max_impl
        pca_df["NROY"] = max_impl <= threshold

        # Create plot
        plt.figure(figsize=(12, 10))

        # Create scatter plot
        sc = plt.scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            c=pca_df["max_implausibility"],
            cmap="viridis_r",
            s=50,
            alpha=0.7,
        )

        # Highlight NROY points
        nroy_points = pca_df[pca_df["NROY"] == True]
        if not nroy_points.empty:
            plt.scatter(
                nroy_points["PC1"],
                nroy_points["PC2"],
                facecolors="none",
                edgecolors="g",
                s=100,
                linewidths=2,
                label="NROY Points",
            )

        # Add labels
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        plt.colorbar(sc, label="Max Implausibility")
        plt.title("PCA Visualization of Parameter Space")
        plt.grid(True, alpha=0.3)

        if not nroy_points.empty:
            plt.legend()

        plt.tight_layout()
        plt.show()

        # Also show the parameter loadings
        plt.figure(figsize=(10, 6))
        loadings = pca.components_.T

        # Plot PC1 loadings
        plt.subplot(1, 2, 1)
        plt.bar(param_names, loadings[:, 0])
        plt.title("PC1 Loadings")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)

        # Plot PC2 loadings
        plt.subplot(1, 2, 2)
        plt.bar(param_names, loadings[:, 1])
        plt.title("PC2 Loadings")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_3d_visualization(self, df, impl_scores):
        """Create a 3D visualization of parameters"""
        threshold = self.threshold_slider.value
        param_x = self.param_x.value
        param_y = self.param_y.value
        param_z = self.param_z.value

        # Calculate max implausibility
        if len(impl_scores.shape) > 1:
            max_impl = np.max(impl_scores, axis=1)
        else:
            max_impl = impl_scores

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Create scatter plot
        scatter = ax.scatter(
            df[param_x],
            df[param_y],
            df[param_z],
            c=max_impl,
            cmap="viridis_r",
            s=30,
            alpha=0.7,
        )

        # Highlight NROY points
        nroy_points = df[max_impl <= threshold]
        if not nroy_points.empty:
            ax.scatter(
                nroy_points[param_x],
                nroy_points[param_y],
                nroy_points[param_z],
                color="green",
                s=50,
                alpha=1.0,
                marker="o",
                label="NROY Points",
            )

        # Add labels
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_zlabel(param_z)

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Implausibility")

        # Add threshold line on colorbar
        cbar.ax.axhline(y=threshold, color="r", linestyle="--")

        plt.title("3D Visualization of Parameters and Implausibility")

        if not nroy_points.empty:
            plt.legend()

        plt.tight_layout()
        plt.show()

    def _plot_implausibility_radar(self, df, impl_scores):
        """Create radar plots showing implausibility for different outputs"""
        # Make sure impl_scores is 2D
        if len(impl_scores.shape) == 1:
            impl_scores = impl_scores.reshape(-1, 1)

        threshold = self.threshold_slider.value
        sample_idx = self.sample_selector.value

        # Make sure sample_idx is within range
        n_samples = len(df)
        if sample_idx >= n_samples:
            sample_idx = n_samples - 1
            self.sample_selector.value = sample_idx

        # Get values for the selected sample
        if sample_idx < len(impl_scores):
            sample_scores = impl_scores[sample_idx]

            # Get sample parameters
            sample_params = df.iloc[sample_idx].copy()

            # Remove non-parameter entries
            for key in ["min_implausibility", "max_implausibility", "NROY"]:
                if key in sample_params:
                    sample_params.pop(key)

            # Set up angles for radar plot
            n_outputs = len(self.output_names)
            angles = np.linspace(0, 2 * np.pi, n_outputs, endpoint=False).tolist()
            angles += angles[:1]  # Close the loop

            # Create extended output names and scores arrays (for closing the loop)
            extended_output_names = self.output_names + [self.output_names[0]]
            extended_scores = sample_scores.tolist() + [sample_scores[0]]

            # Create figure
            plt.figure(figsize=(12, 8), subplot_kw=dict(polar=True))

            # Plot threshold
            plt.plot(
                angles, [threshold] * len(angles), "r--", linewidth=1, label="Threshold"
            )

            # Fill the area below threshold
            plt.fill(angles, [threshold] * len(angles), "r", alpha=0.1)

            # Plot implausibility
            plt.plot(angles, extended_scores, "b-", linewidth=2, label="Implausibility")
            plt.fill(angles, extended_scores, "b", alpha=0.1)

            # Set labels
            plt.xticks(angles[:-1], self.output_names, fontsize=10)

            # Set y limits
            max_score = max(max(extended_scores), threshold * 1.5)
            plt.ylim(0, max_score)

            # Create title with parameter values
            param_str = "\n".join(
                [f"{key}={value:.3g}" for key, value in sample_params.items()]
            )

            plt.title(
                f"Implausibility Radar for Sample {sample_idx}\n{param_str}",
                fontsize=12,
            )
            plt.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

            # Also show a bar chart with the same data
            plt.figure(figsize=(12, 6))
            bars = plt.bar(self.output_names, sample_scores, alpha=0.7)

            # Color bars based on threshold
            for i, bar in enumerate(bars):
                if sample_scores[i] > threshold:
                    bar.set_color("r")
                else:
                    bar.set_color("g")

            plt.axhline(
                y=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}"
            )
            plt.xlabel("Output")
            plt.ylabel("Implausibility")
            plt.title(f"Implausibility for Sample {sample_idx}")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.text(
                0.5,
                0.5,
                "Sample index out of range",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    def display(self):
        """Display the dashboard"""
        display(self.main_layout)
        # Initialize the first plot
        self._update_plot(None)
