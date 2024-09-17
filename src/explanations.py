import lime
import lime.lime_tabular
import numpy as np
import shap


def get_shap_explanation(model, X_sample):
    """Global SHAP explanation."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample)

def get_lime_explanation(model, X_sample, feature_names):
    """Local LIME explanation."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_sample),
        feature_names=feature_names,
        class_names=['Not churn', 'Churn'],
        mode='classification'
    )
    explanation = explainer.explain_instance(X_sample[0], model.predict_proba)
    explanation.show_in_notebook()

