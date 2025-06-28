"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import utils

def run_digit_classification():
    """Main function to run the complete digit classification pipeline"""
    digits = utils.load_and_view()
    data, target = utils.flatten_images(digits)
    clf = utils.create_classifier(gamma=0.001)
    X_train, X_test, y_train, y_test = utils.split_data(data, target, test_size=0.5)
    predicted = utils.train_and_predict(clf, X_train, y_train, X_test)
    utils.plot_predictions(X_test, predicted)
    utils.print_classification_report(y_test, predicted, clf)
    cm = utils.plot_confusion_matrix(y_test, predicted)
    utils.rebuild_classification_report_from_cm(cm)
    return y_test, predicted, cm

###############################################################################
# Run the complete digit classification pipeline
if __name__ == "__main__":
    run_digit_classification()
    plt.show()
