import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_and_view(n_rows=1, n_cols=4):
    digits = datasets.load_digits()
    
    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    return digits

def flatten_images(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def create_classifier(gamma=0.001):
    return svm.SVC(gamma=gamma)

def split_data(data, target, test_size=0.5, random_state=None):
    return train_test_split(data, target, test_size=test_size, random_state=random_state)

def train_and_predict(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return predicted

def plot_predictions(X_test, predicted, n_rows=1, n_cols=4):
    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

def print_classification_report(y_test, predicted, clf):
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

def plot_confusion_matrix(y_test, predicted):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return disp.confusion_matrix

def rebuild_classification_report_from_cm(confusion_matrix):
    from sklearn import metrics
    y_true = []
    y_pred = []
    
    for gt in range(len(confusion_matrix)):
        for pred in range(len(confusion_matrix)):
            y_true += [gt] * confusion_matrix[gt][pred]
            y_pred += [pred] * confusion_matrix[gt][pred]
    
    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )
