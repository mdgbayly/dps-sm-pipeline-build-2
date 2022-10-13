import argparse
import os
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
import joblib


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train logistic regression on sparse feature matrix.')
	parser.add_argument('--X_train_file', type=str)
	parser.add_argument('--X_test_file', type=str)
	parser.add_argument('--user_type', type=str)
	parser.add_argument('--iter', type=int, default=1000)
	args = parser.parse_args()

	features_suffix = (args.X_train_file.split("-")[-1]).split(".")[0]

	train_file = os.path.join('/opt/ml/processing/train', args.X_train_file)
	print(f'train_file: ${train_file}')
	test_file = os.path.join('/opt/ml/processing/test', args.X_test_file)
	print(f'test_file: ${test_file}')

	# Load sparse dataset
	train = csr_matrix(load_npz(train_file))

	# First column is the label
	X_train, y_train = train[:, 1:], train[:, 0].toarray().flatten()

	# Train
	model = LogisticRegression(solver="lbfgs", max_iter=args.iter)
	model.fit(X_train, y_train)

	joblib.dump(model, os.path.join('/opt/ml/processing/model', 'model.joblib'))

	X_test = csr_matrix(load_npz(test_file))
	y_pred_test = model.predict_proba(X_test)[:, 1]

	np.save(os.path.join('/opt/ml/processing/model', f"y-pred-test-{args.user_type}-{features_suffix}"), y_pred_test)
