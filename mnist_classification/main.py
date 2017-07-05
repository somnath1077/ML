from mnist_classification.data_loader import load_training_data, load_test_data
from mnist_classification.evaluate_predictions import report
from mnist_classification.train_predict import train_model, predict

train_data, train_labels = load_training_data()
test_data, test_labels = load_test_data()

classifier = train_model(train_data, train_labels, hidden_layer_size=(100, 50))
predictions = predict(classifier, test_data)
pred_report = report(predictions, test_labels)
print(pred_report)
