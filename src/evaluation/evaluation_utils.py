def evaluate_uncertainty_attributions(
    uncertainty_attributions,
    metric_list,
    uq_model,
    unpacked_dataset,
    base_model,
    ensemble,
    pred_tests,
    explainer,
    uq_strategy,
    mc_passes,
    nr_testsamples,
    model_props,
    X_test,
    empirical_xuq_generator,
):
    """evaluate the generated uncertainty attributions w.r.t. the specified metrics

    Args:
        uncertainty_attributions: uncertainty_attributions
        metric_list: list of metric objects
        uq_model: base trained model with uq
        unpacked_dataset: tuple of split dataset
        base_model: base trained model without uq
        ensemble: list of torch models
        pred_tests: test set predictions
        explainer: xuq explanation generator
        uq_strategy: uncertainty quantification strategy
        mc_passes: Monte Carlo passes
        nr_testsamples: Number of test samples
        model_props: Model properties
        X_test: test set
        empirical_xuq_generator: empirical uncertainty attributions generator (used for the OOD and RIS Metrics)

    Returns:
        Tuple(dict, dict): A tuple containing two dictionaries, one with aggregated metric values and one with all values, for example, for complexity, the first dict includes the average complexity over all samples, while the second dict includes the complexity value for each sample
    """
    metric_values = {}

    for metric in metric_list:
        metrics, values = metric.evaluate_uncertainty_attributions(
            uncertainty_attributions=uncertainty_attributions,
            uq_model=uq_model,
            unpacked_dataset=unpacked_dataset,
            explainer=explainer,
            uq_strategy=uq_strategy,
            pred_tests=pred_tests,
            base_model=base_model,
            ensemble=ensemble,
            mc_passes=mc_passes,
            nr_testsamples=nr_testsamples,
            model_props=model_props,
            X_test=X_test,
            empirical_xuq_generator=empirical_xuq_generator,
        )
        metric_values[metric.get_name()] = metrics
        metric_values[metric.get_name()]["all_values"] = values
    return metric_values
