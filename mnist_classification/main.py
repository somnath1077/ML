from mnist_classification.data_loader import load_training_data, load_test_data, append_report_file
from mnist_classification.evaluate_predictions import report
from mnist_classification.train_predict import train_model, predict

train_data, train_labels = load_training_data()
test_data, test_labels = load_test_data()
HIDDEN_LAYER_CONFIG = [(400, 200, 100, 10), (400, 300, 100, 10)]

for hidden_layer in HIDDEN_LAYER_CONFIG:
    classifier = train_model(train_data, train_labels, hidden_layer_size=hidden_layer)
    predictions = predict(classifier, test_data)
    pred_report = report(predictions, test_labels)
    heading = "Hidden Layer Size = {} \n".format(hidden_layer)
    append_report_file(pred_report, heading)
