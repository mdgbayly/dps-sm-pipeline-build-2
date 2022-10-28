import os
import argparse
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timezone
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
import joblib
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class TimeWindowQueue:
	"""A queue for counting efficiently the number of events within time windows.
	Complexity:
		All operators in amortized O(W) time where W is the number of windows.

	From JJ's KTM repository: https://github.com/jilljenn/ktm.
	"""

	def __init__(self, window_lengths):
		self.queue = []
		self.window_lengths = window_lengths
		self.cursors = [0] * len(self.window_lengths)

	def __len__(self):
		return len(self.queue)

	def get_counters(self, t):
		self.update_cursors(t)
		return [len(self.queue)] + [len(self.queue) - cursor for cursor in self.cursors]

	def push(self, time):
		self.queue.append(time)

	def push_many(self, timestamps):
		self.queue.extend(timestamps)

	def update_cursors(self, t):
		for pos, length in enumerate(self.window_lengths):
			while (self.cursors[pos] < len(self.queue) and
				   t - self.queue[self.cursors[pos]] >= length):
				self.cursors[pos] += 1


def phi(x):
	return np.log(1 + x)


WINDOW_LENGTHS = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
NUM_WINDOWS = len(WINDOW_LENGTHS) + 1


def reformat_train_entries(entries):
	interactions = []
	for entry in entries:
		for i, t in enumerate(entry['timestamp']):
			interactions.append({
				'user_id': entry['sk'],
				'item_id': entry['pk'],
				'timestamp': int(t),
				'correct': int(entry['correct'][i])
			})
	return interactions


def process_user_training(features, q_mat_dict, user_id, entries, counters):
	print('processing user entries')

	interactions = reformat_train_entries(entries)

	df = pd.DataFrame.from_records(interactions)
	df = df.sort_values(by=['timestamp'])
	df = df.drop_duplicates()

	df_user = df.values
	num_items_user = df_user.shape[0]

	# TODO Include qmatrix later
	# skills = Q_mat[df_user[:, 1].astype(int)].copy()
	skills = np.empty((0, 0))
	num_items, num_skills = Q_mat.shape

	features['df'] = np.vstack((features['df'], df_user))

	labels = df_user[:, 3].reshape(-1, 1).astype(int)
	features['labels'] = np.vstack((features['labels'], labels))

	# Current skills one hot encoding
	if 's' in active_features:
		features['s'] = sparse.vstack([features["s"], sparse.csr_matrix(skills)])

	# Attempts
	if 'a' in active_features:
		# Time windows
		if 'tw' in active_features:
			attempts = np.zeros((num_items_user, (num_skills + 2) * NUM_WINDOWS))

			for i, (item_id, ts) in enumerate(df_user[:, 1:3]):
				# Past attempts for relevant skills
				if 'sc' in active_features:
					for skill_id in q_mat_dict[item_id]:
						counts = phi(np.array(counters[user_id, skill_id, "skill"].get_counters(ts)))
						attempts[i, skill_id * NUM_WINDOWS:(skill_id + 1) * NUM_WINDOWS] = counts
						counters[user_id, skill_id, "skill"].push(ts)

				# Past attempts for item
				if 'ic' in active_features:
					counts = phi(np.array(counters[user_id, item_id, "item"].get_counters(ts)))
					attempts[i, -2 * NUM_WINDOWS:-1 * NUM_WINDOWS] = counts
					counters[user_id, item_id, "item"].push(ts)

				# Past attempts for all items
				if 'tc' in active_features:
					counts = phi(np.array(counters[user_id].get_counters(ts)))
					attempts[i, -1 * NUM_WINDOWS:] = counts
					counters[user_id].push(ts)

		features['a'] = sparse.vstack([features['a'], sparse.csr_matrix(attempts)])

	# Wins
	if "w" in active_features:
		# Time windows
		if 'tw' in active_features:
			wins = np.zeros((num_items_user, (num_skills + 2) * NUM_WINDOWS))

			for i, (item_id, ts, correct) in enumerate(df_user[:, 1:4]):
				# Past wins for relevant skills
				if 'sc' in active_features:
					for skill_id in q_mat_dict[item_id]:
						counts = phi(np.array(counters[user_id, skill_id, "skill", "correct"].get_counters(ts)))
						wins[i, skill_id * NUM_WINDOWS:(skill_id + 1) * NUM_WINDOWS] = counts
						if correct:
							counters[user_id, skill_id, "skill", "correct"].push(ts)

				# Past wins for item
				if 'ic' in active_features:
					counts = phi(np.array(counters[user_id, item_id, "item", "correct"].get_counters(ts)))
					wins[i, -2 * NUM_WINDOWS:-1 * NUM_WINDOWS] = counts
					if correct:
						counters[user_id, item_id, "item", "correct"].push(ts)

				# Past wins for all items
				if 'tc' in active_features:
					counts = phi(np.array(counters[user_id, "correct"].get_counters(ts)))
					wins[i, -1 * NUM_WINDOWS:] = counts
					if correct:
						counters[user_id, "correct"].push(ts)

		features['w'] = sparse.vstack([features['w'], sparse.csr_matrix(wins)])


def reformat_inference_entries(entries):
	interactions = []
	for entry in entries:
		interactions.append({
			'user_id': entry['sk'],
			'item_id': entry['pk'],
			'timestamp': [ int(t) for t in entry['timestamp']],
			'correct': [int(c) for c in entry['correct']]
		})
	return interactions


def process_user_inferences(features, q_mat_dict, user_id, entries, counters):

	print('processing user inferences')

	interactions = reformat_inference_entries(entries)
	df = pd.DataFrame.from_records(interactions)

	df_user = df.values
	num_items_user = df_user.shape[0]

	# TODO Include qmatrix later
	# skills = Q_mat[df_user[:, 1].astype(int)].copy()
	skills = np.empty((0, 0))
	num_items, num_skills = Q_mat.shape

	features['df'] = np.vstack((features['df'], df_user[:, [0, 1]]))

	ts_now = int(datetime.now(timezone.utc).timestamp())

	# Current skills one hot encoding
	if 's' in active_features:
		features['s'] = sparse.vstack([features["s"], sparse.csr_matrix(skills)])

	# Attempts
	if 'a' in active_features:
		# Time windows
		if 'tw' in active_features:
			attempts = np.zeros((num_items_user, (num_skills + 2) * NUM_WINDOWS))

			for i, (item_id) in enumerate(df_user[:, 1]):

				# Past attempts for relevant skills
				if 'sc' in active_features:
					for skill_id in q_mat_dict[item_id]:
						counts = phi(np.array(counters[user_id, skill_id, "skill"].get_counters(ts_now)))
						attempts[i, skill_id * NUM_WINDOWS:(skill_id + 1) * NUM_WINDOWS] = counts

				# Past attempts for item
				if 'ic' in active_features:
					counts = phi(np.array(counters[user_id, item_id, "item"].get_counters(ts_now)))
					attempts[i, -2 * NUM_WINDOWS:-1 * NUM_WINDOWS] = counts

				# Past attempts for all items
				if 'tc' in active_features:
					counts = phi(np.array(counters[user_id].get_counters(ts_now)))
					attempts[i, -1 * NUM_WINDOWS:] = counts

		features['a'] = sparse.vstack([features['a'], sparse.csr_matrix(attempts)])

	# Wins
	if "w" in active_features:
		# Time windows
		if 'tw' in active_features:
			wins = np.zeros((num_items_user, (num_skills + 2) * NUM_WINDOWS))

			for i, (item_id) in enumerate(df_user[:, 1]):
				# Past wins for relevant skills
				if 'sc' in active_features:
					for skill_id in q_mat_dict[item_id]:
						counts = phi(np.array(counters[user_id, skill_id, "skill", "correct"].get_counters(ts_now)))
						wins[i, skill_id * NUM_WINDOWS:(skill_id + 1) * NUM_WINDOWS] = counts

				# Past wins for item
				if 'ic' in active_features:
					counts = phi(np.array(counters[user_id, item_id, "item", "correct"].get_counters(ts_now)))
					wins[i, -2 * NUM_WINDOWS:-1 * NUM_WINDOWS] = counts

				# Past wins for all items
				if 'tc' in active_features:
					counts = phi(np.array(counters[user_id, "correct"].get_counters(ts_now)))
					wins[i, -1 * NUM_WINDOWS:] = counts

		features['w'] = sparse.vstack([features['w'], sparse.csr_matrix(wins)])


def process_user(features, q_mat_dict, user_id, entries):
	if user_id is None:
		return

	# Counters for continuous time windows
	counters = defaultdict(lambda: TimeWindowQueue(WINDOW_LENGTHS))

	process_user_training(features['training'], q_mat_dict, user_id, entries, counters)

	process_user_inferences(features['inference'], q_mat_dict, user_id, entries, counters)


def init_skills_attempts(features, num_skills):
	# Skill features
	if 's' in active_features:
		features["s"] = sparse.csr_matrix(np.empty((0, num_skills)))

	# Past attempts and wins features
	for key in ['a', 'w']:
		if key in active_features:
			if 'tw' in active_features:
				features[key] = sparse.csr_matrix(np.empty((0, (num_skills + 2) * NUM_WINDOWS)))
			else:
				features[key] = sparse.csr_matrix(np.empty((0, num_skills + 2)))


def init_features(num_skills):
	features = {
		'training': {},
		'inference': {}
	}
	# Keep track of original features and labels
	features['training']['df'] = np.empty((0, 4))
	features['training']['labels'] = np.empty((0, 1), dtype=int)
	features['inference']['df'] = np.empty((0, 2))

	init_skills_attempts(features['training'], num_skills)
	init_skills_attempts(features['inference'], num_skills)

	return features


def one_hot_user(one_hot_encoder, features):
	features['training']['u'] = one_hot_encoder.fit_transform(features['training']["df"][:, 0].reshape(-1, 1))
	features['inference']['u'] = one_hot_encoder.transform(features['inference']["df"][:, 0].reshape(-1, 1))


def one_hot_item(one_hot_encoder, features):
	features['training']['i'] = one_hot_encoder.fit_transform(features['training']["df"][:, 1].reshape(-1, 1))
	features['inference']['i'] = one_hot_encoder.transform(features['inference']["df"][:, 1].reshape(-1, 1))


def save_features(user_type, features):
	if features['training']['df'].shape[0] == 0:
		print(f'No features for user type {user_type}')
		return

	# User and item one hot encodings
	one_hot = OneHotEncoder(handle_unknown='ignore')
	if 'u' in active_features:
		one_hot_user(one_hot, features)

	if 'i' in active_features:
		one_hot_item(one_hot, features)

	features['training'].pop('df')
	df_user_practices = features['inference'].pop('df')
	x_train = sparse.hstack([sparse.csr_matrix(features['training']['labels']),
							sparse.hstack([features['training'][x] for x in features['training'].keys() if x != 'labels'])]).tocsr()
	x_test = sparse.hstack([sparse.csr_matrix(features['inference']['u']),
							sparse.hstack([features['inference'][x] for x in features['inference'].keys() if x != 'u'])]).tocsr()

	sparse.save_npz(os.path.join('/opt/ml/processing/train', f"X-train-{user_type}-{encode_features_suffix}"), x_train)
	sparse.save_npz(os.path.join('/opt/ml/processing/test', f"X-test-{user_type}-{encode_features_suffix}"), x_test)
	np.save(os.path.join('/opt/ml/processing/test', f"X-test-{user_type}-user-practices"), df_user_practices)

def update_pipeline_count(table):
	try:
		table.update_item(
			Key= {
				'pk': 'CONTROL1',
				'sk': 'PIPELINE'
			},
			UpdateExpression='SET tally = tally + :inc_1, updatedAt = :updated_at',
			ExpressionAttributeValues={
				':inc_1': 1,
				':updated_at': f"{datetime.utcnow().isoformat()[:-3]}Z"
			}
		)
	except ClientError as err:
		print(
			"Couldn't update pipeline control count Here's why: %s: %s",
			err.response['Error']['Code'],
			err.response['Error']['Message'])
		raise


def default_values(entry):
	if 'timestamp' not in entry:
		entry['timestamp'] = []
	
	if 'correct' not in entry:
		entry['correct'] = []
	
	return entry


def process_practices_step(endpoint_url):
	session = boto3.Session()

	dynamodb = session.resource(
		'dynamodb', endpoint_url=endpoint_url, region_name='us-east-1'
	)

	practices_table = dynamodb.Table('dps_dev_practice')

	update_pipeline_count(practices_table)

	# TODO Include skill matrix later
	num_items, num_skills = Q_mat.shape

	# Transform q-matrix into dictionary for fast lookup
	Q_mat_dict = {i: set() for i in range(num_items)}
	for i, j in np.argwhere(Q_mat == 1):
		Q_mat_dict[i].add(j)

	features = {
		'G': init_features(num_skills),
		'U': init_features(num_skills)
	}

	query_kwargs = {
		'IndexName': "gsi-2",
		'KeyConditionExpression': Key('gsi_2_pk').eq("1")
	}
	try:
		done = False
		start_key = None
		current_user = None
		user_entries = []
		while not done:
			if start_key:
				query_kwargs['ExclusiveStartKey'] = start_key
			response = practices_table.query(**query_kwargs)

			for r in response.get('Items', []):
				user = r['sk']
				new_user = user != current_user
				if new_user:
					if current_user is not None:
						process_user(features[current_user[0]], Q_mat_dict, current_user, user_entries)
						user_entries = []
					current_user = user

				user_entries.append(default_values(r))

			start_key = response.get('LastEvaluatedKey', None)
			done = start_key is None
		process_user(features[current_user[0]], Q_mat_dict, current_user, user_entries)
	except ClientError as err:
		print(
			"Couldn't scan for practices. Here's why: %s: %s",
			err.response['Error']['Code'],
			err.response['Error']['Message'])
		raise

	save_features('G', features['G'])
	save_features('U', features['U'])


def train_eval_step(args):
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


THRESHOLD = 0.3


def update_item_user_rpd(table, practice, user, prediction):
	rpd = str(prediction - THRESHOLD)
	try:
		table.update_item(
			Key= {
				'pk': practice,
				'sk': user
			},
			UpdateExpression='SET gsi_1_sk = :rpd',
			ExpressionAttributeValues={
				':rpd': rpd
			}
		)
	except ClientError as err:
		print(
			"Couldn't update practice Here's why: %s: %s",
			err.response['Error']['Code'],
			err.response['Error']['Message'])
		raise


def update_rpd(table, user_practice, prediction):
	query_kwargs = {
		'KeyConditionExpression': Key('pk').eq(user_practice[1]) & Key('sk').begins_with(user_practice[0])
	}
	try:
		done = False
		start_key = None
		while not done:
			if start_key:
				query_kwargs['ExclusiveStartKey'] = start_key
			response = table.query(**query_kwargs)

			for r in response.get('Items', []):
				practice = r['pk']
				user = r['sk']
				update_item_user_rpd(table, practice, user, prediction)

			start_key = response.get('LastEvaluatedKey', None)
			done = start_key is None
	except ClientError as err:
		print(
			"Couldn't scan for practices. Here's why: %s: %s",
			err.response['Error']['Code'],
			err.response['Error']['Message'])
		raise


def update_probabilities(endpoint_url, practices, predictions):
	session = boto3.Session()
	dynamodb = session.resource(
		'dynamodb', endpoint_url=endpoint_url, region_name='us-east-1'
	)

	practices_table = dynamodb.Table('dps_dev_practice')

	for i, practice in enumerate(practices):
		update_rpd(practices_table, practice, predictions[i])


def update_rpd_step(args):
	pred_file = os.path.join('/opt/ml/processing/model', args.y_predictions)
	print(f'pred_file: {pred_file}')
	user_practices_file = os.path.join('/opt/ml/processing/test', args.X_user_practices)
	print(f'user_practices_file: {user_practices_file}')

	X_practices = np.load(user_practices_file, allow_pickle=True)
	y_predictions = np.load(pred_file, allow_pickle=True)

	update_probabilities(args.endpoint_url, X_practices, y_predictions)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Encode sparse feature matrix for logistic regression.')
	# Encode arguments
	parser.add_argument('--endpoint_url', type=str)
	parser.add_argument('-u', action='store_true',
						help='If True, include user one hot encoding.')
	parser.add_argument('-i', action='store_true',
						help='If True, include item one hot encoding.')
	parser.add_argument('-s', action='store_true',
						help='If True, include skills many hot encoding .')
	parser.add_argument('-ic', action='store_true',
						help='If True, include item historical counts.')
	parser.add_argument('-sc', action='store_true',
						help='If True, include skills historical counts.')
	parser.add_argument('-tc', action='store_true',
						help='If True, include total historical counts.')
	parser.add_argument('-w', action='store_true',
						help='If True, historical counts include wins.')
	parser.add_argument('-a', action='store_true',
						help='If True, historical counts include attempts.')
	parser.add_argument('-tw', action='store_true',
						help='If True, historical counts are encoded as time windows.')

	# Train Eval arguments
	parser.add_argument('--X_train_file', type=str)
	parser.add_argument('--X_test_file', type=str)
	parser.add_argument('--user_type', type=str)
	parser.add_argument('--iter', type=int, default=1000)
	
	# Update RPD arguments
	parser.add_argument('--X_user_practices', type=str)
	parser.add_argument('--y_predictions', type=str)

	args = parser.parse_args()

	all_features = ['u', 'i', 's', 'ic', 'sc', 'tc', 'w', 'a', 'tw']
	active_features = [features for features in all_features if vars(args)[features]]
	encode_features_suffix = ''.join(active_features)

	# TODO Include Q Matrix later
	Q_mat = np.empty((0, 0))

	try:
		print('*** Encoding Step ***')
		process_practices_step(args.endpoint_url)

		print('*** Training Step ***')
		train_eval_step(args)

		print('*** Update RPD Step ***')
		update_rpd_step(args)

	except Exception as Argument:
		logger.exception("Pipeline failed")
