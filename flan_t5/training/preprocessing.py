import boto3
import botocore
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Optional

# Replace these values with your own
access_key_id = os.environ['S3_ACCESS_KEY_ID']
secret_access_key = os.environ['S3_SECRET_ACCESS_KEY']
bucket_name = os.environ['S3_BUCKET_NAME']
target = os.environ['DATA_PATH']  # Replace with the actual object key

def s3_download(destination: str=''):
# Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    try:
        # Download the dataset
        s3.download_file(bucket_name, target, destination)
        print(f"Dataset downloaded to './{destination}'")
    except botocore.exceptions.NoCredentialsError:
        print("AWS credentials not found or invalid.")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

def construct_prompt(prompt_template: str=None, context: str=None):
    return f'{prompt_template}\n\nContext: \n{context}'

def load_hf_dataset(file_path: str='', download: Optional[bool]=True):
    path = os.path.join(os.getcwd(), file_path)
    print(path)
    if download and not os.path.exists(path):
        s3_download(destination=path)
        data = pd.read_csv(os.path.join(path)).sample(n=1000)
        # return Dataset.from_pandas(pd.read_csv(os.path.join(path)))
        return Dataset.from_pandas(data)
    else:
        data = pd.read_csv(os.path.join(path)).sample(n=1000)
        # return Dataset.from_pandas(pd.read_csv(os.path.join(path)))
        return Dataset.from_pandas(data)

class Preprocessor:
    def __init__(
            self, 
            prompt_template: str='', 
            model_name: str='google/flan-t5-small', 
            max_source_length: int=517, 
            max_target_length: int=128,
            input_column_name: str='patent_text',
            target_column_name: str='subclass_id'
        ):
        self.model_name=model_name
        self.max_source_length=max_source_length
        self.max_target_length=max_target_length
        self.prompt_template=prompt_template
        self.input=input_column_name
        self.target=target_column_name

    def preprocess(self, sample, padding='max_length'):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = [construct_prompt(self.prompt_template, item) for item in sample[self.input]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=self.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample[self.target], max_length=self.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

if __name__ == '__main__':

    prompt_template = 'Generate USPC labels for the following text: '
    model_name = 'google/flan-t5-large'
    max_source_length = 517
    max_target_length = 128
    input_column_name='patent_text'
    target_column_name='subclass_id'
    save_dataset_path = os.path.join(os.getcwd(), 'data')

    dataset = load_hf_dataset(file_path='data/grouped_labels.csv')
    processed_data = dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)
    print("test run with 1000 examples")
    preprocessor=Preprocessor(
        prompt_template, 
        model_name, 
        max_source_length, 
        max_target_length,
        input_column_name,
        target_column_name
    )
    train_tokenized_dataset = processed_data['train'].map(preprocessor.preprocess, batched=True, remove_columns=list(processed_data['train'].features))
    test_tokenized_dataset = processed_data['test'].map(preprocessor.preprocess, batched=True, remove_columns=list(processed_data['test'].features))
    train_tokenized_dataset.save_to_disk(os.path.join(save_dataset_path,"train"))
    test_tokenized_dataset.save_to_disk(os.path.join(save_dataset_path,"test"))



