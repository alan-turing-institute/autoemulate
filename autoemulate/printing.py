from autoemulate.utils import get_mean_scores
from autoemulate.utils import get_model_name
from autoemulate.utils import get_model_scores


def print_cv_results(models, scores_df, model=None, sort_by="r2", param_search=False):
    """Print cv results.

    Parameters
    ----------
    models : list
        A list of models.
    scores_df : pandas.DataFrame
        A dataframe with scores for each model, metric, and fold.
    model : str, optional
        The name of the model to print. If None, the best fold from each model will be printed.
        If a model name is provided, the scores for that model across all folds will be printed.
    sort_by : str, optional
        The metric to sort by. Default is "r2".

    """
    # check if model is in self.models
    if model is not None:
        model_names = [get_model_name(model) for model in models]
        if model not in model_names:
            raise ValueError(
                f"Model {model} not found. Available models are: {model_names}"
            )
    if model is None:
        if param_search:
            means = get_mean_scores(scores_df, metric=sort_by)
            print("Test score for each model:")
            print(means)
        else:
            means = get_mean_scores(scores_df, metric=sort_by)
            print("Average scores across all models:")
            print(means)
    else:
        if param_search:
            scores = get_model_scores(scores_df, model)
            print(f"Test score for {model}:")
            print(scores)
        else:
            scores = get_model_scores(scores_df, model)
            print(f"Scores for {model} across all folds:")
            print(scores)
