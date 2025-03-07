import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.metaestimators import _BaseComposition


class OutputOnlyPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for sklearn Pipeline that applies preprocessing
    to only the y (output) values during model training.

    This preprocessor can be used to apply dimensionality reduction (like PCA)
    or other transformations to target variables before model training.

    Parameters
    ----------
    methods : str or list of str, default=None
        The preprocessing method(s) to apply to y.
        Options include:
        - 'pca': Apply PCA to reduce dimensionality
        - 'standardize': Apply StandardScaler
        - custom methods can be added

    n_components : int or float, default=None
        For PCA, the number of components to keep.
        If None and method includes 'pca', keeps all components.

    Attributes
    ----------
    transformers_ : dict
        Dictionary of fitted transformer objects with method names as keys.
    inverse_transformers_ : dict
        Dictionary of inverse transformation methods for each transformer.
    """

    def __init__(self, methods=None, n_components=None):
        self.methods = methods
        self.n_components = n_components
        self.transformers_ = {}
        self.inverse_transformers_ = {}

    def fit(self, X, y=None):
        """
        Fit all the transformers on the output data.

        Parameters
        ----------
        X : array-like
            Input features (not used in this transformer)
        y : array-like
            Target values to be preprocessed

        Returns
        -------
        self : object
            Returns self.
        """
        if y is None:
            return self

        # Ensure y is a numpy array
        y = np.asarray(y)

        # Initialize transformers based on methods
        if self.methods is None:
            return self

        methods = self.methods if isinstance(self.methods, list) else [self.methods]

        for method in methods:
            if method == "pca":
                transformer = PCA(n_components=self.n_components)
                self.transformers_[method] = transformer.fit(y)
                self.inverse_transformers_[
                    method
                ] = lambda y_transformed, transformer=transformer: transformer.inverse_transform(
                    y_transformed
                )

            elif method == "standardize":
                transformer = StandardScaler()
                self.transformers_[method] = transformer.fit(y)
                self.inverse_transformers_[
                    method
                ] = lambda y_transformed, transformer=transformer: transformer.inverse_transform(
                    y_transformed
                )

        return self

    def transform(self, X, y=None):
        """
        Transform input data X (pass-through) and output data y (if provided).
        During fitting in a pipeline, y is provided and transformed.

        Parameters
        ----------
        X : array-like
            Input features (returned unchanged)
        y : array-like, default=None
            Target values to transform

        Returns
        -------
        X : array-like
            Original input features
        y_transformed : array-like or None
            Transformed target values (if y was provided)
        """
        # During pipeline.fit(), return X unchanged and transformed y
        if y is not None and self.methods is not None:
            y = np.asarray(y)
            methods = self.methods if isinstance(self.methods, list) else [self.methods]

            y_transformed = y.copy()
            for method in methods:
                if method in self.transformers_:
                    y_transformed = self.transformers_[method].transform(y_transformed)

            return X, y_transformed

        # During pipeline.transform() or pipeline.predict(), return X unchanged
        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : array-like
            Input features
        y : array-like, default=None
            Target values

        Returns
        -------
        X : array-like
            Original input features
        y_transformed : array-like or None
            Transformed target values (if y was provided)
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform_y(self, y_transformed):
        """
        Inverse transform the output data.

        Parameters
        ----------
        y_transformed : array-like
            Transformed target values

        Returns
        -------
        y : array-like
            Original target values
        """
        if self.methods is None or not self.transformers_:
            return y_transformed

        methods = self.methods if isinstance(self.methods, list) else [self.methods]
        y = y_transformed.copy()

        # Apply inverse transformations in reverse order
        for method in reversed(methods):
            if method in self.inverse_transformers_:
                y = self.inverse_transformers_[method](y)

        return y


class OutputOnlyPreprocessortwo(BaseEstimator, TransformerMixin):
    """
    Custom transformer for preprocessing only output (Y) data.
    Can apply PCA or other dimensionality reduction techniques to Y.
    """

    def __init__(self, methods=None):
        """
        Initialize the preprocessor with methods to apply to Y.

        Parameters:
        -----------
        methods : list or single sklearn transformer
            Transformer(s) to apply to Y. Each transformer should implement fit_transform and inverse_transform.
        """
        self.methods = methods if isinstance(methods, list) else [methods]
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit the Y preprocessor with y data.
        X is ignored as this transformer only processes Y.

        Parameters:
        -----------
        X : array-like
            Input features (ignored by this transformer)
        y : array-like
            Target values to be transformed

        Returns:
        --------
        self : object
            Returns self
        """
        if y is None:
            return self

        y_reshaped = self._reshape_y(y)

        for method in self.methods:
            if method is not None:
                method.fit(y_reshaped)

        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Return X unchanged as this transformer only processes Y.

        Parameters:
        -----------
        X : array-like
            Input features to be returned unchanged

        Returns:
        --------
        X : array-like
            The input X is returned unchanged
        """
        return X

    def fit_transform(self, X, y=None):
        """
        Fit to y and return X unchanged.

        Parameters:
        -----------
        X : array-like
            Input features to be returned unchanged
        y : array-like
            Target values to fit on

        Returns:
        --------
        X : array-like
            The input X is returned unchanged
        """
        self.fit(X, y)
        return self.transform(X)

    def transform_y(self, y):
        """
        Apply the transformation to y data.

        Parameters:
        -----------
        y : array-like
            Target values to transform

        Returns:
        --------
        y_transformed : array-like
            Transformed target values
        """
        if not self.is_fitted_:
            raise ValueError(
                "This OutputOnlyPreprocessor instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

        if y is None:
            return None

        y_reshaped = self._reshape_y(y)
        y_transformed = y_reshaped.copy()

        for method in self.methods:
            if method is not None:
                y_transformed = method.transform(y_transformed)

        # Return with the original shape structure
        if len(y_transformed.shape) > len(y.shape):
            return y_transformed.squeeze()
        return y_transformed

    def inverse_transform_y(self, y):
        """
        Apply the inverse transformation to y data.

        Parameters:
        -----------
        y : array-like
            Transformed target values to inverse transform

        Returns:
        --------
        y_original : array-like
            Inverse transformed target values
        """
        if not self.is_fitted_:
            raise ValueError(
                "This OutputOnlyPreprocessor instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

        if y is None:
            return None

        y_reshaped = self._reshape_y(y)
        y_original = y_reshaped.copy()

        for method in reversed(self.methods):
            if method is not None:
                y_original = method.inverse_transform(y_original)

        # Return with the original shape structure
        if len(y_original.shape) > len(y.shape):
            return y_original.squeeze()
        return y_original

    def _reshape_y(self, y):
        """
        Reshape y to 2D if needed for sklearn transformers.

        Parameters:
        -----------
        y : array-like
            Target values to reshape

        Returns:
        --------
        y_reshaped : array-like
            Reshaped target values
        """
        y_array = np.asarray(y)
        if y_array.ndim == 1:
            return y_array.reshape(-1, 1)
        return y_array


class YPreprocessingPipeline(_BaseComposition):
    """
    Pipeline that can preprocess both X and Y data.
    """

    def __init__(self, y_preprocessor=None, x_pipeline=None):
        """
        Initialize the Y-preprocessing pipeline.

        Parameters:
        -----------
        y_preprocessor : OutputOnlyPreprocessor
            Preprocessor for Y data
        x_pipeline : sklearn.pipeline.Pipeline
            Standard pipeline for X data
        """
        self.y_preprocessor = y_preprocessor
        self.x_pipeline = x_pipeline

    def fit(self, X, y=None):
        """
        Fit the pipeline with both X and Y preprocessing.

        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target values

        Returns:
        --------
        self : object
            Returns self
        """
        if y is not None and self.y_preprocessor is not None:
            # First fit the Y preprocessor
            self.y_preprocessor.fit(X, y)
            # Transform Y
            y_transformed = self.y_preprocessor.transform_y(y)
        else:
            y_transformed = y

        # Fit the standard pipeline
        if self.x_pipeline is not None:
            self.x_pipeline.fit(X, y_transformed)

        return self

    def predict(self, X):
        """
        Predict using the pipeline and reverse any Y transformations.

        Parameters:
        -----------
        X : array-like
            Input features

        Returns:
        --------
        y_pred : array-like
            Predictions with any Y preprocessing reversed
        """
        # Use the X pipeline to predict
        if self.x_pipeline is not None:
            y_pred = self.x_pipeline.predict(X)
        else:
            raise ValueError("No X pipeline specified for prediction.")

        # Reverse any Y transformations
        if self.y_preprocessor is not None:
            y_pred = self.y_preprocessor.inverse_transform_y(y_pred)

        return y_pred

    def score(self, X, y):
        """
        Score the pipeline with Y preprocessing.

        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target values

        Returns:
        --------
        score : float
            Score value
        """
        # Transform the true y values
        if self.y_preprocessor is not None:
            y_transformed = self.y_preprocessor.transform_y(y)
        else:
            y_transformed = y

        if self.x_pipeline is not None:
            return self.x_pipeline.score(X, y_transformed)
        else:
            raise ValueError("No X pipeline specified for scoring.")
