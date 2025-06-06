import corner
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from IPython.display import display
from scipy.stats import gaussian_kde


class MCMCVisualizationDashboard:
    """
    Interactive dashboard for MCMC calibration results visualization.
    Provides dynamic plot selection and parameter filtering capabilities.
    """

    def __init__(self, calibrator, threshold=3.0):
        """
        Initialize dashboard with MCMC calibrator results.

        Parameters:
        -----------
        calibrator : MCMCCalibrator
            The fitted MCMC calibrator with results
        threshold : float
            Default threshold for various analyses
        """
        if not hasattr(calibrator, "mcmc_results"):
            raise ValueError("Calibrator must have run MCMC first")

        self.calibrator = calibrator
        self.results = calibrator.mcmc_results
        self.param_names = list(self.results.keys())
        self.threshold = threshold

        # Convert results to DataFrame for easier handling
        self.samples_df = pd.DataFrame(self.results)

        # Calculate summary statistics
        self._calculate_summary_stats()

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create the UI elements
        self._create_ui()

    def _calculate_summary_stats(self):
        """Calculate summary statistics for the MCMC results."""
        if hasattr(self.calibrator, "mcmc_summary"):
            self.summary = self.calibrator.mcmc_summary
        else:
            # Create basic summary if not available
            summary_data = []
            for param in self.param_names:
                samples = self.results[param]
                summary_data.append(
                    {
                        "Parameter": param,
                        "Mean": np.mean(samples),
                        "Std": np.std(samples),
                        "Q2.5": np.percentile(samples, 2.5),
                        "Q50": np.percentile(samples, 50),
                        "Q97.5": np.percentile(samples, 97.5),
                    }
                )
            self.summary = pd.DataFrame(summary_data)

    def _create_ui(self):
        """Create the user interface widgets with dynamic controls"""

        # Plot type selection
        self.plot_type = widgets.Dropdown(
            options=[
                ("Corner Plot", "corner"),
                ("Trace Plots", "traces"),
                ("Posterior Distributions", "posteriors"),
                ("Correlation Heatmap", "correlations"),
                ("Pairwise Plots", "pairwise"),
                ("Convergence Diagnostics", "convergence"),
                ("Parameter Evolution", "evolution"),
                ("Posterior vs Prior", "vs_prior"),
                ("Summary Statistics", "summary"),
            ],
            value="corner",
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

        # Figure size controls
        self.fig_width = widgets.IntSlider(
            value=12,
            min=2,
            max=15,
            step=2,
            description="Width:",
            style={"description_width": "initial"},
        )

        self.fig_height = widgets.IntSlider(
            value=10,
            min=2,
            max=15,
            step=2,
            description="Height:",
            style={"description_width": "initial"},
        )

        # Plot-specific options
        self.n_bins = widgets.IntSlider(
            value=30,
            min=10,
            max=60,
            step=5,
            description="Bins:",
            style={"description_width": "initial"},
        )

        self.show_kde = widgets.Checkbox(
            value=True, description="Show KDE", style={"description_width": "initial"}
        )

        self.show_prior = widgets.Checkbox(
            value=True, description="Show Prior", style={"description_width": "initial"}
        )

        # Sample size for performance
        self.sample_size = widgets.IntText(
            value=1000,
            description="Sample Size:",
            style={"description_width": "initial"},
        )

        # Burnin control
        self.burnin = widgets.IntSlider(
            value=0,
            min=0,
            max=len(list(self.results.values())[0]) // 4,
            step=10,
            description="Burn-in:",
            style={"description_width": "initial"},
        )

        # Create parameter checkboxes for multi-parameter plots
        self.param_checkboxes = []
        for param in self.param_names:
            cb = widgets.Checkbox(
                value=True, description=param, disabled=False, indent=False
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

        # Plot explanation box
        self.plot_explanation = widgets.HTML(
            value="<div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; border-left: 4px solid #2196F3;'><b>Corner Plot:</b> Shows joint and marginal posterior distributions. Diagonal shows individual parameter distributions, off-diagonal shows parameter correlations.</div>",
            layout=widgets.Layout(width="400px", margin="10px 0px"),
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
        self.param_selectors = widgets.HBox([self.param_x, self.param_y])
        self.size_controls = widgets.HBox([self.fig_width, self.fig_height])
        self.plot_controls = widgets.HBox([self.n_bins])
        self.advanced_controls = widgets.HBox([self.show_kde, self.show_prior])

        # Container for parameter selection controls
        self.param_selection_controls = widgets.VBox(
            [self.param_selection_label, self.param_checkbox_container]
        )

        # Container for parameter selection + explanation (side by side)
        self.param_and_explanation = widgets.HBox(
            [self.param_selection_controls, self.plot_explanation]
        )

        # Container for corner plot specific controls
        self.corner_controls = widgets.VBox([self.sample_size])

        # Container for trace plot specific controls
        self.trace_controls = widgets.VBox([self.burnin])

        # Main controls that are always visible
        controls_top = widgets.HBox([self.plot_type])
        controls_size = self.size_controls

        self.main_layout = widgets.VBox(
            [
                controls_top,
                controls_size,
                self.param_selectors,
                self.plot_controls,
                self.advanced_controls,
                self.param_and_explanation,  # Use combined container
                self.corner_controls,
                self.trace_controls,
                self.update_button,
                self.output,
            ]
        )

        # Initially hide plot-specific controls
        self._update_visible_controls({"new": "corner"})

    def _update_visible_controls(self, change):
        """Show/hide controls based on selected plot type"""
        plot_type = change["new"]

        # Update plot explanation based on plot type
        explanations = {
            "corner": "<div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; border-left: 4px solid #2196F3;'><b>Corner Plot:</b> Shows joint and marginal posterior distributions. Diagonal shows individual parameter distributions, off-diagonal shows parameter correlations and dependencies.</div>",
            "traces": "<div style='background-color: #f0f8e8; padding: 10px; border-radius: 5px; border-left: 4px solid #4CAF50;'><b>Trace Plots:</b> Shows MCMC chain evolution over iterations. Good traces should look like 'white noise' around a stable mean. Use burn-in to remove initial non-converged samples.</div>",
            "posteriors": "<div style='background-color: #fff3e0; padding: 10px; border-radius: 5px; border-left: 4px solid #FF9800;'><b>Posterior Distributions:</b> Shows individual parameter posterior distributions as histograms with optional KDE smoothing. Compare with uniform priors to see parameter learning.</div>",
            "correlations": "<div style='background-color: #fce4ec; padding: 10px; border-radius: 5px; border-left: 4px solid #E91E63;'><b>Correlation Heatmap:</b> Shows linear correlations between all parameters. Values close to ±1 indicate strong correlations, near 0 indicates independence.</div>",
            "pairwise": "<div style='background-color: #f3e5f5; padding: 10px; border-radius: 5px; border-left: 4px solid #9C27B0;'><b>Pairwise Plots:</b> Comprehensive view with scatter plots (upper), KDE contours (lower), and histograms (diagonal). Shows both linear and non-linear relationships.</div>",
            "convergence": "<div style='background-color: #e0f2f1; padding: 10px; border-radius: 5px; border-left: 4px solid #009688;'><b>Convergence Diagnostics:</b> Running means (top) should stabilize, autocorrelations (bottom) should decay quickly. Helps assess if MCMC has converged properly.</div>",
            "evolution": "<div style='background-color: #e8eaf6; padding: 10px; border-radius: 5px; border-left: 4px solid #3F51B5;'><b>Parameter Evolution:</b> Shows normalized parameter traces on same plot. Useful for comparing convergence rates and identifying problematic parameters.</div>",
            "vs_prior": "<div style='background-color: #fff8e1; padding: 10px; border-radius: 5px; border-left: 4px solid #FFC107;'><b>Posterior vs Prior:</b> Compares learned posteriors with original priors. Large differences indicate strong data influence; similar shapes suggest weak data information.</div>",
            "summary": "<div style='background-color: #efebe9; padding: 10px; border-radius: 5px; border-left: 4px solid #795548;'><b>Summary Statistics:</b> Displays tabular summary of posterior statistics including means, standard deviations, credible intervals, and convergence diagnostics.</div>",
        }

        self.plot_explanation.value = explanations.get(
            plot_type, explanations["corner"]
        )

        # Default - show basic parameter selectors
        self.param_selectors.layout.display = "flex"
        self.plot_controls.layout.display = "flex"
        self.advanced_controls.layout.display = "none"

        # Hide all conditional controls by default
        self.param_and_explanation.layout.display = "none"
        self.corner_controls.layout.display = "none"
        self.trace_controls.layout.display = "none"

        # Show controls based on plot type
        if plot_type == "corner":
            # Corner plot: show parameter selection and corner-specific controls
            self.param_and_explanation.layout.display = "flex"
            self.corner_controls.layout.display = "flex"
            self.param_selectors.layout.display = "none"  # Use checkboxes instead

        elif plot_type == "traces":
            # Trace plots: show parameter selection and burnin control
            self.param_and_explanation.layout.display = "flex"
            self.trace_controls.layout.display = "flex"
            self.param_selectors.layout.display = "none"

        elif plot_type == "posteriors":
            # Posterior plots: show parameter selection and advanced controls
            self.param_and_explanation.layout.display = "flex"
            self.advanced_controls.layout.display = "flex"
            self.param_selectors.layout.display = "none"

        elif plot_type in ["correlations", "summary"]:
            # These don't need parameter selection
            self.param_selectors.layout.display = "none"
            self.plot_controls.layout.display = "none"

        elif plot_type in ["pairwise", "evolution"]:
            # These use parameter selection checkboxes
            self.param_and_explanation.layout.display = "flex"
            self.param_selectors.layout.display = "none"

        elif plot_type == "convergence":
            # Convergence diagnostics
            self.param_and_explanation.layout.display = "flex"
            self.param_selectors.layout.display = "none"

        elif plot_type == "vs_prior":
            # Prior comparison
            self.param_and_explanation.layout.display = "flex"
            self.advanced_controls.layout.display = "flex"
            self.param_selectors.layout.display = "none"

    def _get_selected_parameters(self):
        """Get list of selected parameters from checkboxes"""
        selected = []
        for i, checkbox in enumerate(self.param_checkboxes):
            if checkbox.value:
                selected.append(self.param_names[i])
        return selected if selected else self.param_names

    def _get_selected_data(self, selected_params=None, apply_burnin=False):
        """Get data for selected parameters with optional burnin"""
        if selected_params is None:
            selected_params = self._get_selected_parameters()

        data = {}
        burnin = self.burnin.value if apply_burnin else 0

        for param in selected_params:
            if param in self.results:
                data[param] = self.results[param][burnin:]

        return data

    def _update_plot(self, _):
        """Update the plot based on current widget values"""
        with self.output:
            clear_output(wait=True)

            try:
                plot_type = self.plot_type.value
                figsize = (self.fig_width.value, self.fig_height.value)

                if plot_type == "corner":
                    self._plot_corner(figsize)
                elif plot_type == "traces":
                    self._plot_traces(figsize)
                elif plot_type == "posteriors":
                    self._plot_posteriors(figsize)
                elif plot_type == "correlations":
                    self._plot_correlations(figsize)
                elif plot_type == "pairwise":
                    self._plot_pairwise(figsize)
                elif plot_type == "convergence":
                    self._plot_convergence(figsize)
                elif plot_type == "evolution":
                    self._plot_evolution(figsize)
                elif plot_type == "vs_prior":
                    self._plot_vs_prior(figsize)
                elif plot_type == "summary":
                    self._show_summary()

                plt.show()

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
                print(f"Detailed error: {e}")
                import traceback

                traceback.print_exc()

    def _plot_corner(self, figsize):
        """Generate corner plot"""
        selected_params = self._get_selected_parameters()

        if len(selected_params) < 2:
            plt.figure(figsize=figsize)
            plt.text(
                0.5,
                0.5,
                "Corner plot requires at least 2 parameters",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.axis("off")
            return

        selected_data = self._get_selected_data(selected_params, apply_burnin=True)
        samples_array = np.column_stack(
            [selected_data[param] for param in selected_params]
        )

        # Subsample if needed
        if self.sample_size.value and len(samples_array) > self.sample_size.value:
            indices = np.random.choice(
                len(samples_array), size=self.sample_size.value, replace=False
            )
            samples_array = samples_array[indices]

        # Use default quantiles
        quantiles = [0.025, 0.5, 0.975]

        fig = corner.corner(
            samples_array,
            labels=selected_params,
            bins=self.n_bins.value,
            smooth=0.9,
            show_titles=True,
            quantiles=quantiles,
            figsize=figsize,
        )

        fig.suptitle("Posterior Parameter Distributions", fontsize=16, y=0.98)

    def _plot_traces(self, figsize):
        """Generate trace plots"""
        selected_params = self._get_selected_parameters()
        selected_data = self._get_selected_data(selected_params)

        n_params = len(selected_params)
        n_cols = min(3, n_params)
        n_rows = int(np.ceil(n_params / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        burnin = self.burnin.value

        for i, param in enumerate(selected_params):
            ax = axes[i]
            samples = selected_data[param]

            # Plot full trace
            ax.plot(samples, linewidth=0.8, color="blue")

            # Highlight burnin period
            if burnin > 0:
                ax.axvspan(0, burnin, alpha=0.3, color="red", label="Burn-in")
                ax.plot(samples[burnin:], linewidth=0.8, color="green")

            ax.set_title(f"{param}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

            # Add horizontal line at mean (post burn-in)
            post_burnin_mean = np.mean(samples[burnin:])
            ax.axhline(
                post_burnin_mean, color="red", linestyle="--", alpha=0.7, label="Mean"
            )
            ax.legend()

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.suptitle("MCMC Trace Plots", fontsize=16, y=0.98)

    def _plot_posteriors(self, figsize):
        """Generate posterior distribution plots"""
        selected_params = self._get_selected_parameters()
        selected_data = self._get_selected_data(selected_params, apply_burnin=True)

        n_params = len(selected_params)
        n_cols = min(3, n_params)
        n_rows = int(np.ceil(n_params / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, param in enumerate(selected_params):
            ax = axes[i]
            samples = selected_data[param]

            # Histogram
            ax.hist(
                samples,
                bins=self.n_bins.value,
                alpha=0.7,
                density=True,
                color="skyblue",
                edgecolor="black",
                label="Posterior",
            )

            # KDE
            if self.show_kde.value:
                kde_est = gaussian_kde(samples)
                x_range = np.linspace(samples.min(), samples.max(), 200)
                ax.plot(x_range, kde_est(x_range), "r-", linewidth=2, label="KDE")

            # Prior (uniform) if available
            if self.show_prior.value and hasattr(self.calibrator, "reduced_bounds"):
                if param in self.calibrator.reduced_bounds:
                    bounds = self.calibrator.reduced_bounds[param]
                    prior_height = 1 / (bounds[1] - bounds[0])
                    ax.axhline(
                        prior_height,
                        color="orange",
                        linestyle="--",
                        linewidth=2,
                        label="Prior (Uniform)",
                    )
                    ax.axvline(bounds[0], color="gray", linestyle=":", alpha=0.5)
                    ax.axvline(bounds[1], color="gray", linestyle=":", alpha=0.5)

            # Statistics
            mean_val = np.mean(samples)
            std_val = np.std(samples)
            ax.axvline(
                mean_val, color="red", linewidth=2, label=f"Mean: {mean_val:.3f}"
            )

            ax.set_title(f"{param}\nMean ± Std: {mean_val:.3f} ± {std_val:.3f}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.suptitle("Posterior Distributions", fontsize=16, y=0.98)

    def _plot_correlations(self, figsize):
        """Generate correlation heatmap"""
        selected_data = self._get_selected_data(apply_burnin=True)

        if len(selected_data) < 2:
            plt.figure(figsize=figsize)
            plt.text(
                0.5,
                0.5,
                "Correlation plot requires at least 2 parameters",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.axis("off")
            return

        # Create DataFrame
        samples_df = pd.DataFrame(selected_data)
        corr_matrix = samples_df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title("Parameter Correlation Matrix", fontsize=16, pad=20)

    def _plot_pairwise(self, figsize):
        """Generate pairwise scatter plots"""
        selected_params = self._get_selected_parameters()
        selected_data = self._get_selected_data(selected_params, apply_burnin=True)

        if len(selected_params) < 2:
            plt.figure(figsize=figsize)
            plt.text(
                0.5,
                0.5,
                "Pairwise plot requires at least 2 parameters",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.axis("off")
            return

        # Subsample if needed
        if self.sample_size.value:
            first_param_len = len(list(selected_data.values())[0])
            if first_param_len > self.sample_size.value:
                indices = np.random.choice(
                    first_param_len, size=self.sample_size.value, replace=False
                )
                selected_data = {
                    param: samples[indices] for param, samples in selected_data.items()
                }

        # Create DataFrame
        df = pd.DataFrame(selected_data)

        # Create pairplot
        g = sns.PairGrid(df, height=figsize[0] / len(selected_params))
        # 1. Add alpha for transparency
        g.map_upper(plt.scatter, s=20, alpha=0.6)

        # 2. Adjust KDE levels for better visibility
        g.map_lower(sns.kdeplot, cmap="Blues", fill=True, alpha=0.7, levels=10)

        # 3. Add correlation coefficients to upper triangle
        def corrfunc(x, y, **kws):
            r = np.corrcoef(x, y)[0, 1]
            ax = plt.gca()
            ax.annotate(
                f"r = {r:.2f}",
                xy=(0.1, 0.9),
                xycoords="axes fraction",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        g.map_upper(corrfunc)
        g.fig.suptitle("Pairwise Parameter Relationships", fontsize=16, y=0.98)

    def _plot_convergence(self, figsize):
        """Generate convergence diagnostics"""
        selected_params = self._get_selected_parameters()
        selected_data = self._get_selected_data(selected_params)

        n_params = len(selected_params)
        fig, axes = plt.subplots(2, n_params, figsize=figsize)

        if n_params == 1:
            axes = axes.reshape(-1, 1)

        for i, param in enumerate(selected_params):
            samples = selected_data[param]

            # Running mean
            ax1 = axes[0, i]
            running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
            ax1.plot(running_mean)
            ax1.axhline(
                np.mean(samples), color="red", linestyle="--", label="Final Mean"
            )
            ax1.set_title(f"{param} - Running Mean")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Running Mean")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Autocorrelation
            ax2 = axes[1, i]
            lags = np.arange(0, min(100, len(samples) // 4))
            autocorr = [
                np.corrcoef(samples[:-lag], samples[lag:])[0, 1] if lag > 0 else 1.0
                for lag in lags
            ]

            ax2.plot(lags, autocorr, "o-", markersize=3)
            ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
            ax2.axhline(
                0.1, color="orange", linestyle=":", alpha=0.5, label="10% threshold"
            )
            ax2.set_title(f"{param} - Autocorrelation")
            ax2.set_xlabel("Lag")
            ax2.set_ylabel("Autocorrelation")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.suptitle("Convergence Diagnostics", fontsize=16, y=0.98)

    def _plot_evolution(self, figsize):
        """Plot parameter evolution over iterations"""
        selected_params = self._get_selected_parameters()
        selected_data = self._get_selected_data(selected_params)

        fig, ax = plt.subplots(figsize=figsize)

        for param in selected_params:
            samples = selected_data[param]
            # Normalize to [0,1] for better comparison
            normalized = (samples - samples.min()) / (samples.max() - samples.min())
            ax.plot(normalized, label=param, alpha=0.7)

        ax.set_title("Parameter Evolution (Normalized)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Normalized Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    def _plot_vs_prior(self, figsize):
        """Plot posterior vs prior comparison"""
        selected_params = self._get_selected_parameters()
        selected_data = self._get_selected_data(selected_params, apply_burnin=True)

        n_params = len(selected_params)
        n_cols = min(2, n_params)
        n_rows = int(np.ceil(n_params / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, param in enumerate(selected_params):
            ax = axes[i]
            samples = selected_data[param]

            # Posterior
            if self.show_kde.value:
                kde_est = gaussian_kde(samples)
                x_range = np.linspace(samples.min(), samples.max(), 200)
                ax.plot(x_range, kde_est(x_range), "b-", linewidth=3, label="Posterior")
            else:
                ax.hist(
                    samples,
                    bins=self.n_bins.value,
                    alpha=0.7,
                    density=True,
                    color="skyblue",
                    label="Posterior",
                )

            # Prior if available
            if (
                hasattr(self.calibrator, "reduced_bounds")
                and param in self.calibrator.reduced_bounds
            ):
                bounds = self.calibrator.reduced_bounds[param]
                prior_height = 1 / (bounds[1] - bounds[0])
                ax.axhline(
                    prior_height,
                    color="red",
                    linestyle="--",
                    linewidth=3,
                    label="Prior (Uniform)",
                )
                ax.axvspan(bounds[0], bounds[1], alpha=0.2, color="red")

            ax.set_title(f"{param}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.suptitle("Posterior vs Prior Comparison", fontsize=16, y=0.98)

    def _show_summary(self):
        """Display summary statistics"""
        print("MCMC Posterior Summary:")
        print("=" * 80)
        print(self.summary.to_string(index=False))

        # Add convergence information if available
        print("\n" + "=" * 80)
        print("Convergence Information:")

        # Calculate effective sample size (simple estimate)
        for param in self.param_names:
            samples = self.results[param]
            # Simple autocorrelation-based ESS estimate
            autocorr = np.correlate(samples, samples, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]
            autocorr = autocorr / autocorr[0]

            # Find first negative autocorrelation
            first_negative = np.where(autocorr < 0)[0]
            if len(first_negative) > 0:
                tau = first_negative[0]
            else:
                tau = len(autocorr)

            ess = len(samples) / (2 * tau + 1)
            print(f"{param}: ESS ≈ {ess:.0f} ({ess/len(samples)*100:.1f}%)")

    def display(self):
        """Display the dashboard"""
        heading = widgets.HTML(value="<h2> MCMC Visualization Dashboard</h2>")

        # Display the heading and instructions first
        display(heading)
        display(self.main_layout)

        # Initialize
