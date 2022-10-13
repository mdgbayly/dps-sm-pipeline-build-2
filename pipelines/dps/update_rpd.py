import argparse
import os
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
import numpy as np

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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Apply recall probabilities to practice entities.')
	parser.add_argument('--endpoint_url', type=str)
	parser.add_argument('--X_user_practices', type=str)
	parser.add_argument('--y_predictions', type=str)

	args = parser.parse_args()

	pred_file = os.path.join('/opt/ml/processing/model', args.y_predictions)
	print(f'pred_file: {pred_file}')
	user_practices_file = os.path.join('/opt/ml/processing/test', args.X_user_practices)
	print(f'user_practices_file: {user_practices_file}')

	X_practices = np.load(user_practices_file, allow_pickle=True)
	y_predictions = np.load(pred_file, allow_pickle=True)

	update_probabilities(args.endpoint_url, X_practices, y_predictions)

