
from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float

if TYPE_CHECKING:
    import napari


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def segment_with_xgboost(
    image: "napari.types.ImageData",
    annotation: "napari.types.LabelsData",
    filename: "str"  = "xgb_model.json",
) -> "napari.types.LabelsData":
    import apoc
    import xgboost as xgb
    from datetime import datetime

    feature_images = apoc.generate_feature_stack(image) # todo: make features configurable

    X_for_training, Y_for_training = _to_np(feature_images, annotation)

    # Train the model
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_for_training, Y_for_training)

    # Apply
    X_for_prediction, _ = _to_np(feature_images, ground_truth=None)
    y_pred = xgb_model.predict(X_for_prediction)
    labels_pred = (y_pred + 1).reshape(annotation.shape)

    return labels_pred


def _to_np(features, ground_truth=None):
    """
    Convert given feature and ground truth images in the right format to be processed by scikit-learn.

    Parameters
    ----------
    features : list of ndarray
    ground_truth : ndarray

    Returns
    -------
    features, ground_truth with each feature and ground_truth a one-dimensional list of numbers

    See Also
    --------
    https://github.com/haesleinhuepf/apoc/blob/main/apoc/_pixel_classifier.py#L231
    """
    import numpy as np
    feature_stack = np.asarray([np.asarray(f).ravel() for f in features]).T
    if ground_truth is None:
        return feature_stack, None
    else:
        # make the annotation 1-dimensional
        ground_truth_np = np.asarray(ground_truth).ravel()

        X = feature_stack
        y = ground_truth_np

        # remove all pixels from the feature and annotations which have not been annotated
        mask = y > 0
        X = X[mask]
        y = y[mask]

        return X, y - 1