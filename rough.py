import json
import boto3
import random
import csv
from collections import defaultdict
import io
import uuid

# Initialize AWS clients
sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')

# S3 and SageMaker configurations
BUCKET_NAME = 'mmchatbot'
TRAINING_FOLDER = 'training-data'
DATASET_FILE = 'dataset.csv'
DATASET_S3_URI = f's3://{BUCKET_NAME}/{TRAINING_FOLDER}/{DATASET_FILE}'

# SageMaker configuration
ROLE_ARN = 'arn:aws:iam::324037306866:role/service-role/MMChatbot'  # Replace with your SageMaker execution role ARN
TRAINING_JOB_NAME_PREFIX = 'image-classification-job'
IMAGE_CLASSIFIER_ALGORITHM = 'image-classification'
s3_dataset_path = "s3://mmchatbot/training-data/dataset.csv"


# Function to split the dataset into training and validation
def split_dataset(csv_file):
    # Read the CSV file and split into train and validation
    train_data = []
    validation_data = []

    with open(csv_file, newline='') as f:
        csvreader = csv.reader(f)
        next(csvreader)  # Skip the header if present
        data = list(csvreader)

        # Shuffle and split into train and validation (80% train, 20% validation)
        random.shuffle(data)
        split_idx = int(0.8 * len(data))  # 80% training, 20% validation
        train_data = data[:split_idx]
        validation_data = data[split_idx:]

    return train_data, validation_data


# Function to get input data channel configuration
def get_input_data_channel(data, channel_name):
    s3_uri = f"s3://{BUCKET_NAME}/training-data/{channel_name}"
    return {
        'ChannelName': channel_name,  # Use simple channel names like 'train' and 'validation'
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': s3_uri,
                'S3DataDistributionType': 'FullyReplicated'
            }
        },
        'ContentType': 'application/x-image'  # Use the correct ContentType for image data
    }


def upload_data_to_s3(data, s3_path):
    # Create a BytesIO object to simulate a file in memory
    with io.StringIO() as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the data to the CSV in memory
        csvwriter.writerows(data)

        # Move the file pointer to the beginning of the file
        csvfile.seek(0)

        # Upload the file object to S3
        s3.upload_fileobj(io.BytesIO(csvfile.getvalue().encode('utf-8')), BUCKET_NAME, s3_path)


# Function to get the number of unique classes and total samples in the dataset
def get_num_classes_and_samples():
    # Download the dataset file from S3
    dataset_url = f"{TRAINING_FOLDER}/{DATASET_FILE}"
    s3.download_file(BUCKET_NAME, dataset_url, '/tmp/dataset.csv')
    print("Data Downloaded")

    # Initialize a set to store unique labels and a counter for samples
    labels = set()
    num_samples = 0

    # Open the CSV file and process the contents
    with open('/tmp/dataset.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the header row (if any)
        next(csvreader, None)

        # Iterate over the rows in the CSV
        for row in csvreader:
            image_path, label = row
            labels.add(label)  # Add label to the set (automatically handles uniqueness)
            num_samples += 1  # Increment the sample count

    # Get the number of unique classes (labels)
    num_classes = len(labels)

    return num_classes, num_samples


def create_lst_file(data, filename):
    lst_file_content = []
    for image_path, label in data:
        lst_file_content.append(f"{image_path} {label}\n")

    # Save the list to a file
    lst_file_path = f"/tmp/{filename}"
    with open(lst_file_path, 'w') as f:
        f.writelines(lst_file_content)

    return lst_file_path


def upload_lst_to_s3(lst_file_path, s3_path):
    s3.upload_file(lst_file_path, BUCKET_NAME, s3_path)


def initiate_training_job():
    dataset_url = f"{TRAINING_FOLDER}/{DATASET_FILE}"
    print("Dataset_url: ", dataset_url)

    # Download the dataset file
    s3.download_file(BUCKET_NAME, dataset_url, '/tmp/dataset.csv')

    # Split the dataset into train and validation sets
    train_data, validation_data = split_dataset('/tmp/dataset.csv')
    print("validation_data: ", validation_data)

    # Create .lst files for train and validation
    train_lst_path = create_lst_file(train_data, "train.lst")
    validation_lst_path = create_lst_file(validation_data, "validation.lst")

    # Upload .lst files to S3
    upload_lst_to_s3(train_lst_path, 'training-data/train/train.lst')
    upload_lst_to_s3(validation_lst_path, 'training-data/validation/validation.lst')

    print("Uploaded .lst files")

    # Define the input channels
    input_data_config = [
        get_input_data_channel('training-data/train/train.lst', 'train'),
        get_input_data_channel('training-data/validation/validation.lst', 'validation')
    ]

    # Get the number of classes and samples dynamically
    num_classes, num_samples = get_num_classes_and_samples()
    print("input_data_config: ", input_data_config)

    # SageMaker training job configuration
    job_id = str(uuid.uuid4())
    training_job_name = f'{TRAINING_JOB_NAME_PREFIX}-{job_id}'

    response = sagemaker.create_training_job(
        TrainingJobName=training_job_name,  # Provide a unique name
        AlgorithmSpecification={
            'TrainingImage': '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',
            # Update with your training image
            'TrainingInputMode': 'File'
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=input_data_config,
        OutputDataConfig={
            'S3OutputPath': f's3://{BUCKET_NAME}/training-data/output/'
        },
        ResourceConfig={
            'InstanceType': 'ml.p2.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600
        },
        HyperParameters={
            'num_classes': str(num_classes),  # Dynamically obtained
            'num_training_samples': str(num_samples)  # Dynamically obtained
        }
    )

    return response


# Lambda function handler
def lambda_handler(event, context):
    print("Lambda function to initiate SageMaker training job started.")

    try:
        # Initiate training job
        response = initiate_training_job()
        print(f"Training job initiated successfully. ARN: {response['TrainingJobArn']}")

        # Return success response
        return {
            'statusCode': 200,
            'body': json.dumps(
                {'message': 'Training job initiated successfully', 'TrainingJobArn': response['TrainingJobArn']})
        }

    except Exception as e:
        print(f"Error initiating training job: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Error initiating training job', 'error': str(e)})
        }
