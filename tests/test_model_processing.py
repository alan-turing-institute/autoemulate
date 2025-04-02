import numpy as np
import pytest
import copy
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from autoemulate.emulators import GaussianProcessSklearn
from autoemulate.emulators import RadialBasisFunctions
from autoemulate.emulators import RandomForest
from autoemulate.emulators import SupportVectorMachines
from autoemulate.model_processing import ModelPrepPipeline
from autoemulate.model_registry import ModelRegistry
from autoemulate.utils import get_model_name


@pytest.fixture
def model_registry():
    model_registry = ModelRegistry()
    model_registry.register_model(
        "RadialBasisFunctions", RadialBasisFunctions, is_core=True
    )
    model_registry.register_model(
        "GaussianProcessSklearn", GaussianProcessSklearn, is_core=True
    )
    model_registry.register_model(
        "SupportVectorMachines", SupportVectorMachines, is_core=True
    )
    return model_registry

@pytest.fixture
def basic_pipeline(model_registry):
    """Fixture providing a basic ModelPrepPipeline instance for testing."""
    # Get model names as a list (keys from the dictionary)
    model_names = list(model_registry.get_model_names().keys())
    y = np.array([[1, 2], [3, 4]])  # 2D output for multioutput testing
    
    return ModelPrepPipeline(
        model_registry=model_registry,
        model_names=model_names,  # Now passing a list of names
        y=y,
        prep_config={"name": "None", "params": {}},
        scale_input=False,
        reduce_dim_input=False,
        scale_output=False,
        reduce_dim_output=False
    )
@pytest.fixture
def pipeline_factory(basic_pipeline):
    """Factory to create modified versions of basic_pipeline."""
    def _make_pipeline(**kwargs):
        # Create a copy of the basic pipeline
        pipeline = copy.deepcopy(basic_pipeline)
        
        # Update attributes
        for key, value in kwargs.items():
            setattr(pipeline, key, value)
        
        # Rebuild the pipeline
        pipeline._wrap_model_reducer_in_pipeline()
        return pipeline
    return _make_pipeline
    

@pytest.fixture(params=[
    {"name": "None", "params": {}},
    {"name": "PCA", "params": {"n_components": 8}}
])

# -----------------------test turning models into multioutput-------------------#

def test_turn_models_into_multioutput(basic_pipeline):
    """Test that single-output models are wrapped in MultiOutputRegressor when y is 2D."""
    models_multi = basic_pipeline.models_multi
    
    assert isinstance(models_multi, list)
    
    for model in models_multi:
        # Handle both wrapped and unwrapped models
        original_model = getattr(model, 'estimator', model)
        if not getattr(original_model, '_more_tags', lambda: {})().get('multioutput', False):
            assert isinstance(model, MultiOutputRegressor), \
                f"Expected {type(model).__name__} to be wrapped in MultiOutputRegressor"

# -----------------------test wrapping models in pipeline-------------------#

def test_wrap_models_in_pipeline_no_scaler(pipeline_factory):
    """Test pipeline without scaler."""
    pipeline = pipeline_factory(
        scale_input=False,
        scaler_input=None,
        reduce_dim_input=False,
        dim_reducer_input=None
    )
    
    for model in pipeline.models_piped:
        if isinstance(model, TransformedTargetRegressor):
            model = model.regressor
        
        assert isinstance(model, Pipeline)
        step_names = [step[0] for step in model.steps]
        assert "scaler" not in step_names

      
def test_wrap_models_in_pipeline_no_scaler(pipeline_factory):
    """Test pipeline without scaler."""
    pipeline = pipeline_factory(
        scale_input=False,
        scaler_input=None,
        reduce_dim_input=False,
        dim_reducer_input=None
    )
    
    for model in pipeline.models_piped:
        if isinstance(model, TransformedTargetRegressor):
            model = model.regressor
        
        assert isinstance(model, Pipeline)
        step_names = [step[0] for step in model.steps]
        assert "scaler" not in step_names


def test_pipeline_with_dim_reduction(basic_pipeline):
    """Test that models are wrapped in pipeline with dim reducer when reduce_dim=True."""
    # Configure pipeline with dimensionality reduction
    basic_pipeline.reduce_dim_input = True
    basic_pipeline.dim_reducer_input = PCA(n_components=2)
    basic_pipeline.scale_input = False  # Ensure scaling is off for this test
    basic_pipeline._wrap_model_reducer_in_pipeline()  # Rebuild pipeline with new config
    
    # Check the wrapped models
    assert isinstance(basic_pipeline.models_piped, list)
    
    # Get the model types from the original models
    model_types = tuple(type(model) for model in basic_pipeline.models)
    
    for model in basic_pipeline.models_piped:
        # Get the actual pipeline (might be inside TransformedTargetRegressor)
        if isinstance(model, TransformedTargetRegressor):
            pipeline = model.regressor
        else:
            pipeline = model
        
        # Check it's a Pipeline
        assert isinstance(pipeline, Pipeline), "Model should be wrapped in Pipeline"
        
        # Verify pipeline steps
        step_names = [step[0] for step in pipeline.steps]
        
        if basic_pipeline.reduce_dim_input and not basic_pipeline.scale_input:
            # When only dim reduction is enabled, it should be first step
            assert step_names[0] == "dim_reducer", "First step should be dim_reducer"
            assert isinstance(pipeline.steps[0][1], PCA), "First transformer should be PCA"
        
        # Verify model is the last step
        last_step = pipeline.steps[-1][1]
        assert isinstance(last_step, (MultiOutputRegressor, *model_types)), \
            f"Last step should be model or MultiOutputRegressor, got {type(last_step)}"

def test_pipeline_with_scaler_and_dim_reducer(basic_pipeline):
    """Test that models are wrapped with both scaler and dim reducer when both are enabled."""
    # Configure pipeline with both scaling and dimensionality reduction
    basic_pipeline.scale_input = True
    basic_pipeline.scaler_input = StandardScaler()
    basic_pipeline.reduce_dim_input = True
    basic_pipeline.dim_reducer_input = PCA(n_components=2)
    basic_pipeline._wrap_model_reducer_in_pipeline()  # Rebuild pipeline with new config
    
    # Check the wrapped models
    assert isinstance(basic_pipeline.models_piped, list)
    
    for model in basic_pipeline.models_piped:
        # Get the actual pipeline (might be inside TransformedTargetRegressor)
        if isinstance(model, TransformedTargetRegressor):
            pipeline = model.regressor
        else:
            pipeline = model
        
        # Check it's a Pipeline
        assert isinstance(pipeline, Pipeline), "Model should be wrapped in Pipeline"
        
        # Verify pipeline steps
        step_names = [step[0] for step in pipeline.steps]
        
        # Should have both scaler and dim_reducer when both are enabled
        assert len(step_names) >= 2, "Pipeline should have at least 2 steps"
        assert step_names[0] == "scaler", "First step should be scaler"
        assert isinstance(pipeline.steps[0][1], StandardScaler), "First transformer should be StandardScaler"
        assert step_names[1] == "dim_reducer", "Second step should be dim_reducer"
        assert isinstance(pipeline.steps[1][1], PCA), "Second transformer should be PCA"
        
        # Verify model is the last step
        assert pipeline.steps[-1][0] == "model", "Last step should be the model"

def pipeline_with_preprocessing(request, model_registry):
    """Fixture that provides pipelines with different preprocessing methods."""
    return ModelPrepPipeline(
        model_registry=model_registry,
        model_names=list(model_registry.get_model_names().keys()),
        y=np.random.rand(10, 3),  # 3 outputs
        prep_config=request.param,
        scale_input=True,
        scaler_input=StandardScaler(),
        reduce_dim_input=False,
        scale_output=True,
        scaler_output=StandardScaler(),
        reduce_dim_output=True if request.param["name"] != "None" else False
    )