import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display

from autoemulate.core.types import NumpyLike, TensorLike


class HistoryMatchingDashboard:
    """
    History Matching Dashboard.

    Interactive dashboard for exploring history matching with UI controls that adapt
    based on selected plot type.
    """

    def __init__(
        self,
        samples: NumpyLike | TensorLike,
        impl_scores: NumpyLike | TensorLike,
        param_names: list[str],
        output_names: list[str],
        threshold: float = 3.0,
    ):
        """
        Initialize the dashboard.

        Parameters
        ----------
        samples: NumpyLike or TensorLike
            Parameter samples.
        impl_scores: NumpyLike or TensorLike
            Implausibility scores.
        param_names: list[str]
            Parameter names.
        output_names: list[str]
            Output names.
        threshold: float
            Implausibility threshold.
        """
        # Convert samples to DataFrame
        if isinstance(samples, np.ndarray):
            self.samples_df = pd.DataFrame(samples, columns=param_names)  # pyright: ignore[reportArgumentType]
        elif isinstance(samples, TensorLike):
            self.samples_df = pd.DataFrame(samples.numpy(), columns=param_names)  # pyright: ignore[reportArgumentType]

        # Store other data
        if isinstance(impl_scores, TensorLike):
            self.impl_scores = impl_scores.numpy()
        else:
            self.impl_scores = impl_scores
        self.param_names = param_names
        self.output_names = output_names
        self.threshold = threshold

        # Calculate minimum implausibility for each sample
        if len(self.impl_scores.shape) > 1:
            self.min_impl = np.min(self.impl_scores, axis=1)
            self.max_impl = np.max(self.impl_scores, axis=1)
        else:
            self.min_impl = self.impl_scores
            self.max_impl = self.impl_scores

        # Add implausibility to DataFrame
        self.samples_df["min_implausibility"] = self.min_impl
        self.samples_df["max_implausibility"] = self.max_impl
        self.samples_df["NROY"] = self.max_impl <= threshold

        # Create the UI elements
        self._create_ui()

    def _create_ui(self):
        """Create the user interface widgets with dynamic controls."""
        # Plot type selection
        self.plot_type = widgets.Dropdown(
            options=[
                "Parameter vs Implausibility",
                "Pairwise Parameters",
                "Implausibility Distribution",
                "Parameter Correlation Heatmap",
                "3D Parameter Visualization",
                "Implausibility Radar",
                # "Bayesian Style Comparison",
            ],
            value="Parameter vs Implausibility",
            description="Plot Type:",
            style={"description_width": "initial"},
        )

        # Add observer to show/hide plot-specific controls
        self.plot_type.observe(self._update_visible_controls, names="value")

        # Parameter selection widgets
        self.param_x = widgets.Dropdown(
            options=self.param_names,
            value=self.param_names[0] if self.param_names else None,
            description="X Parameter:",
            disabled=False,
        )

        self.param_y = widgets.Dropdown(
            options=self.param_names,
            value=(
                self.param_names[1]
                if len(self.param_names) > 1
                else self.param_names[0]
            ),
            description="Y Parameter:",
            disabled=False,
        )

        self.param_z = widgets.Dropdown(
            options=self.param_names,
            value=(
                self.param_names[2]
                if len(self.param_names) > 2
                else self.param_names[0]
            ),
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

        # Create parameter checkboxes
        self.param_checkboxes = []
        for param in self.param_names:
            cb = widgets.Checkbox(
                value=True,  # Default all selected
                description=param,
                disabled=False,
                indent=False,
            )
            self.param_checkboxes.append(cb)

        # Group checkboxes in a container with scroll
        self.param_checkbox_container = widgets.VBox(
            self.param_checkboxes,
            layout=widgets.Layout(
                width="auto", height="200px", overflow_y="auto", border="1px solid #ddd"
            ),
        )

        # Label for the checkbox group
        self.param_selection_label = widgets.Label("Select Parameters to Display:")

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

        # Group controls for selective display
        self.param_selectors = widgets.HBox([self.param_x, self.param_y, self.param_z])
        # Container for the parameter selection controls
        self.param_selection_controls = widgets.VBox(
            [self.param_selection_label, self.param_checkbox_container]
        )

        self.radar_controls = widgets.HBox([self.sample_selector])

        # Main controls that are always visible
        controls_top = widgets.HBox([self.plot_type, self.threshold_slider])
        controls_bottom = widgets.HBox([self.nroy_filter])

        self.main_layout = widgets.VBox(
            [
                controls_top,
                self.param_selectors,
                self.radar_controls,
                controls_bottom,
                self.update_button,
                self.output,
            ]
        )

        # Initially hide plot-specific controls
        self.radar_controls.layout.display = "none"
        self.param_z.layout.display = "none"  # Initially hide Z parameter (only for 3D)
        self.param_selection_controls.layout.display = "none"  # Initially hidden

    def _update_visible_controls(self, change: dict):
        """Show/hide controls based on selected plot type."""
        plot_type = change["new"]

        # Default - show X and Y parameters, hide Z parameter
        self.param_x.layout.display = "inline-flex"
        self.param_y.layout.display = "inline-flex"
        self.param_z.layout.display = "none"
        self.param_selection_controls.layout.display = "none"

        # Hide all conditional controls by default
        self.radar_controls.layout.display = "none"

        # Default - hide NROY filter (hide for all plots initially)
        self.nroy_filter.layout.display = "none"

        # Show controls based on plot type
        if plot_type == "3D Parameter Visualization":
            # Show all three parameters for 3D
            self.param_z.layout.display = "inline-flex"
            # Show NROY filter for 3D viz
            self.nroy_filter.layout.display = "flex"

        elif plot_type == "Implausibility Radar":
            # Show sample selector for radar plot
            self.radar_controls.layout.display = "flex"

        elif plot_type in [
            "Parameter Correlation Heatmap",
            "Implausibility Distribution",
            # "Bayesian Style Comparison",
        ]:
            # Hide parameter selectors for plots that don't use them
            if plot_type in [
                "Parameter Correlation Heatmap",
                "Implausibility Distribution",
            ]:
                self.param_x.layout.display = "none"
                self.param_y.layout.display = "none"
            # Don't show NROY filter for these plots

        elif plot_type in [
            "Parameter vs Implausibility",
            "Pairwise Parameters",
            "Emulator Diagnostics",
        ]:
            # Show NROY filter only for these specific plots
            self.nroy_filter.layout.display = "flex"

    def _update_plot(self, _):
        """Update the plot based on current widget values."""
        with self.output:
            clear_output(wait=True)

            # Get current plot type
            plot_type = self.plot_type.value

            filtered_df = self.samples_df.copy()
            filtered_scores = self.impl_scores.copy()

            # Apply NROY filter if selected
            if self.nroy_filter.value:
                # NROY values are boolean
                nroy_mask = filtered_df["NROY"]
                filtered_df = filtered_df[nroy_mask].copy()

                # Filter implausibility scores accordingly
                if len(filtered_scores) == len(filtered_df.index.to_list()):
                    filtered_scores = filtered_scores[
                        np.array(nroy_mask.values, dtype=bool)
                    ].copy()

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
            assert isinstance(filtered_df, pd.DataFrame)
            assert isinstance(filtered_scores, NumpyLike)
            try:
                if plot_type == "Parameter vs Implausibility":
                    self._plot_parameter_vs_implausibility(filtered_df, filtered_scores)
                elif plot_type == "Pairwise Parameters":
                    self._plot_pairwise_parameters(filtered_df, filtered_scores)
                elif plot_type == "Implausibility Distribution":
                    self._plot_implausibility_distribution(filtered_scores)
                elif plot_type == "Parameter Correlation Heatmap":
                    self._plot_parameter_correlation(filtered_df)
                elif plot_type == "3D Parameter Visualization":
                    self._plot_3d_visualization(filtered_df, filtered_scores)
                elif plot_type == "Implausibility Radar":
                    self._plot_implausibility_radar(filtered_df, filtered_scores)
                # elif plot_type == "Bayesian Style Comparison":
                #     self._plot_bayesian_style_comparison(filtered_df, filtered_scores)
                plt.show()

            except Exception as e:
                plt.figure(figsize=(10, 6))
                plt.text(
                    0.5,
                    0.5,
                    f"Error generating plot: {e!s}",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                plt.axis("off")
                plt.tight_layout()
                plt.show()

    def _plot_parameter_vs_implausibility(
        self, df: pd.DataFrame, impl_scores: NumpyLike
    ):
        """Plot parameter vs implausibility."""
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

    def _plot_pairwise_parameters(self, df: pd.DataFrame, impl_scores: NumpyLike):
        """Plot pairwise parameter visualization."""
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
        nroy_points = df[df["NROY"]]
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
            x_min, x_max = nroy_points[param_x].min(), nroy_points[param_x].max()
            y_min, y_max = nroy_points[param_y].min(), nroy_points[param_y].max()
            assert isinstance(x_min, float)
            assert isinstance(x_max, float)
            assert isinstance(y_min, float)
            assert isinstance(y_max, float)
            plt.axvline(x=x_min, color="g", linestyle="--", alpha=0.5)
            plt.axvline(x=x_max, color="g", linestyle="--", alpha=0.5)
            plt.axhline(y=y_min, color="g", linestyle="--", alpha=0.5)
            plt.axhline(y=y_max, color="g", linestyle="--", alpha=0.5)

        plt.title(f"Parameters {param_x} vs {param_y}")
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.colorbar(sc, label="Max Implausibility")
        plt.grid(True, alpha=0.3)
        if not nroy_points.empty:
            plt.legend()
        plt.tight_layout()

    def _plot_implausibility_distribution(self, impl_scores: NumpyLike):
        """Plot implausibility distribution."""
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

            below_threshold = (max_impl <= threshold).sum() / len(max_impl) * 100
        else:
            # Single output case
            plt.hist(impl_scores, bins=30, alpha=0.7, label="Implausibility")
            below_threshold = (impl_scores <= threshold).sum() / len(impl_scores) * 100

        # Add vertical line for threshold
        plt.axvline(
            x=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}"
        )

        plt.title(
            f"Implausibility Distribution\n"
            f"{below_threshold:.1f}% of points below threshold"
        )
        plt.xlabel("Implausibility")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def _plot_parameter_correlation(self, df):
        """Plot parameter correlation heatmap."""
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

    def _plot_3d_visualization(self, df: pd.DataFrame, impl_scores: NumpyLike):
        """Create a 3D visualization of parameters."""
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
            s=30,  # pyright: ignore[reportCallIssue]
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
                s=50,  # pyright: ignore[reportCallIssue]
                alpha=1.0,
                marker="o",
                label="NROY Points",
            )

        # Add labels
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_zlabel(param_z)  # pyright: ignore[reportAttributeAccessIssue]

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Implausibility")

        # Add threshold line on colorbar
        cbar.ax.axhline(y=threshold, color="r", linestyle="--")

        plt.title("3D Visualization of Parameters and Implausibility")

        if not nroy_points.empty:
            plt.legend()

        plt.tight_layout()

    def _plot_implausibility_radar(self, df: pd.DataFrame, impl_scores: NumpyLike):
        """Create radar plots showing implausibility for different outputs."""
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

            # Create extended scores array (for closing the loop)
            extended_scores = [*sample_scores.tolist(), sample_scores[0]]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"polar": True})

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
            max_score = max(*extended_scores, threshold * 1.5)
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

    # def _plot_bayesian_style_comparison(
    #    self, df: pd.DataFrame, impl_scores: NumpyLike
    # ):
    #     """
    #     Create a Bayesian-style visualization showing parameter constraints
    #     with prior and posterior using existing dashboard controls.

    #     This matches the style shown in the example image with:
    #     - Gray shaded prior distributions
    #     - Blue histogram posterior (NROY) distributions
    #     - Support for LaTeX formatted parameter labels
    #     """
    #     import numpy as np

    #     # Calculate max implausibility for each sample
    #     if len(impl_scores.shape) > 1:
    #         max_impl = np.max(impl_scores, axis=1)
    #     else:
    #         max_impl = impl_scores

    #     # Get threshold for NROY classification
    #     threshold = self.threshold_slider.value

    #     # Create NROY indicator (these are our "posterior" samples)
    #     nroy_mask = max_impl <= threshold

    #     # Get the selected parameters from existing UI controls
    #     selected_params = [self.param_x.value, self.param_y.value]

    #     # Remove duplicates while preserving order
    #     selected_params = list(dict.fromkeys(selected_params))

    #     # Create the figure
    #     n_params = len(selected_params)
    #     n_cols = min(2, n_params)
    #     n_rows = (n_params + n_cols - 1) // n_cols

    #     fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    #     # Set the title
    #     title = "History Matching Results for Parameters"
    #     subtitle = "(using NROY points as posterior)"
    #     full_title = f"{title}\n{subtitle}"

    #     # Set the overall title if we have multiple plots
    #     if n_params > 1:
    #         fig.suptitle(full_title, fontsize=16, y=0.98)

    #     # Function to create nice parameter labels with LaTeX
    #     def format_param_label(param):
    #         # Format the parameter name nicely for display
    #         if "log" in param.lower():
    #             base_name = param.replace("log_", "").replace("log", "")
    #             return rf"$\mu_{{{base_name}}}$"
    #         if "_" in param:
    #             parts = param.split("_")
    #             if len(parts) == 2:
    #                 return f"$log_{{10}}({parts[0]}_{{v}}/{parts[0]}_{{h}})$"
    #             return param
    #         return param

    #     # Plot each parameter
    #     for i, param in enumerate(selected_params):
    #         ax = fig.add_subplot(n_rows, n_cols, i + 1)

    #         # Get the prior range (all samples)
    #         param_min = df[param].min()
    #         param_max = df[param].max()

    #         # Add padding
    #         padding = 0.1 * (param_max - param_min)
    #         param_min -= padding
    #         param_max += padding

    #         # Get the posterior data (NROY points)
    #         posterior_data = df.loc[nroy_mask, param]

    #         # Create bins
    #         bins = np.linspace(param_min, param_max, 20).tolist()

    #         # Plot prior (flat uniform distribution)
    #         prior_height = 0.4  # Height for the prior bar
    #         ax.fill_between(
    #             [param_min, param_max],
    #             [0, 0],
    #             [prior_height, prior_height],
    #             color="lightgray",
    #             alpha=0.5,
    #             label="Prior",
    #         )

    #         # Plot posterior
    #         if len(posterior_data) > 0:
    #             ax.hist(
    #                 posterior_data,
    #                 bins=bins,
    #                 density=True,
    #                 alpha=0.7,
    #                 color="royalblue",
    #                 label="Posterior",
    #             )

    #         # Set labels and limits
    #         ax.set_xlabel(format_param_label(param))
    #         ax.set_ylabel("Frequency")
    #         assert isinstance(param_min, float)
    #         assert isinstance(param_max, float)
    #         ax.set_xlim(param_min, param_max)

    #         # Show legend on first plot only
    #         if i == 0:
    #             ax.legend()

    #         # Set title only for single plot
    #         if n_params == 1:
    #             ax.set_title(full_title)

    #     plt.tight_layout()
    #     if n_params > 1:
    #         plt.subplots_adjust(top=0.9)  # Make room for suptitle

    def display(self):
        """Display the dashboard."""
        heading = widgets.HTML(value="<h2>History Matching Dashboard</h2>")

        # Display the heading and instructions first
        display(heading)
        display(self.main_layout)
        # Initialize the first plot
        self._update_plot(None)
