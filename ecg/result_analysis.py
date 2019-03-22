import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.utils.fixes import signature

lw = 2


def plot_roc(Y_test, Y_score, class_num, Type):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(Y_score.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(5, 5))
    plt.plot(fpr[class_num], tpr[class_num], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_num])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    plt.xlabel('Specificity')
    # plt.ylabel('True Positive Rate')
    plt.ylabel('Sensitivity')
    plt.title('Class ' + Type[class_num])
    plt.legend(loc="lower right")
    plt.show()

    return 0


def plot_auc(Y_test, Y_score, class_num, Type):
    precision = dict()
    recall = dict()
    average_precision = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(Y_score.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], Y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], Y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), Y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, Y_score, average="micro")

    plt.figure(figsize=(5, 5))
    plt.plot(recall[class_num], precision[class_num], color='blue',
             lw=lw, label='PRC curve (AP = {0:0.2f})'.format(average_precision[class_num]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Sensitivity (Recall)')
    plt.ylabel('PPV (Precision)')
    plt.title('Class ' + Type[class_num])
    plt.legend(loc="lower left")
    plt.show()

    return 0


def plot_auc2(Y_test, Y_score, class_num, Type):
    precision = dict()
    recall = dict()
    average_precision = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(Y_score.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], Y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], Y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), Y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, Y_score, average="micro")

    plt.figure(figsize=(5, 5))

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall[class_num], precision[class_num], color='b', alpha=0.2,
             where='post', label='PRC curve (AP = {0:0.2f})'.format(average_precision[class_num]))
    plt.fill_between(recall[class_num], precision[class_num], alpha=0.2, color='b', **step_kwargs)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Sensitivity (Recall)')
    plt.ylabel('PPV (Precision)')
    plt.title('Class ' + Type[class_num])
    plt.legend(loc="lower left")
    plt.show()

    return 0


def plot_confusion_matrix(Y_test, Y_score, labels_name, title):

    cm = confusion_matrix(np.argmax(Y_test, 1), np.argmax(Y_score, 1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cmap = plt.get_cmap('Blues')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=45)
    plt.yticks(num_local, labels_name)
    plt.ylabel('MIT-BIH Label')
    plt.xlabel('DNN Predicted Label')
    plt.show()
    return 0






































