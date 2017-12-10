# Service configure
# Author: Zex Li <top_zlynch@yahoo.com>
from os import environ as env

AWS_ACCESS_KEY = env.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = env.get('AWS_SECRET_KEY')
AWS_REGION = env.get('AWS_REGION')
AWS_S3_BUCKET = env.get('AWS_S3_BUCKET')
AWS_SQS_INPUT = env.get('AWS_SQS_INPUT')
AWS_SQS_OUTPUT = env.get('AWS_SQS_OUTPUT')
