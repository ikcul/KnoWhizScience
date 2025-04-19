#!/usr/bin/env python3
# coding: utf-8

import boto3
import botocore
import os

class S3Helper(object):
    def __init__(self):
        os.environ['AWS_PROFILE'] = "KnoWhiz-S3-Access"
        os.environ['AWS_DEFAULT_REGION'] = "us-west-2"

    def download(self, filename:str, output_name:str, bucket_name:str="knowhiz-dev"):
        s3 = boto3.resource('s3')

        try:
            s3.Bucket(bucket_name).download_file(filename, output_name)
            # Construct the full S3 file path
            s3_file_path = f"s3://{bucket_name}/{filename}"
            print(f"The url is: {s3_file_path}, filename is {filename}")
            return s3_file_path
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print(f"The object does not exist.")
            else:
                raise

if __name__ == "__main__":
    s3 = S3Helper()
    s3.download("materials/attention.pdf", "local.pdf", bucket_name="knowhiz-dev")
