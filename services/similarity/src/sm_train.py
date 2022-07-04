import os
import argparse
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

# to pass credentials to into Sagemaker env vars
from dotenv import load_dotenv
load_dotenv()


# THESE ARE METRIC DEFINITIONS FOR SAGEMAKER LOGS
# MODIFY AS NEEDED
metric_def_binary = [
  {"Name": "loss:train", "Regex": "loss: ([0-9\\.]+)"},
  {'Name': 'loss:val', 'Regex': 'val_loss: ([0-9\\.]+)'}
]


# MODIFY YOUR CLI PARAMS AS NEEDED
def parse_args():
    parser = argparse.ArgumentParser()

    # SM job config params
    parser.add_argument('--instance_type', type=str, default='ml.g4dn.4xlarge')
    parser.add_argument('--instance_count', type=int, default=1)
    parser.add_argument('--profile_name', type=str, default='lw2134')
    parser.add_argument('--volume_size', type=int, default=30)
    parser.add_argument('--output_path', type=str, default='s3://magicml-models.dev')
    parser.add_argument('--job_name', type=str, default='magicml-LM')
    
    # train.py params
    parser.add_argument('--model_name', type=str, default='magicml-LM')
    parser.add_argument('--steps', type=int, default=85000)
    parser.add_argument('--batch_size_gpl', type=int, default=32)
    parser.add_argument('--batch_size_qgen', type=int, default=10)
    
    # SM required directories w/ defaults - DO NOT CHANGE
    parser.add_argument('--train', type=str)
    parser.add_argument('--output_dir', type=str, default='/opt/ml/output/data')
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    
    return parser.parse_known_args()


# MODIFY BELOW AS NEEDED
def get_estimator(
  job_name,
  instance_type,
  instance_count,
  profile_name,
  volume_size,
  output_path,
  model_dir,
  model_name,
  steps,
  batch_size_gpl,
  batch_size_qgen
):
  # SAGEMAKER JOB PERMISSIONS AS DEFINED BY YOUR AWS CLI PROFILE
  sess = sagemaker.Session(boto_session=boto3.session.Session(profile_name=profile_name))
  
  # THIS IS FOR DISTRIBUTED TRAINING - EXPERIMENTAL
  if instance_count > 1:
    distribution = {
      'parameter_server': {
        'enabled': True
      }
    }
  else:
    distribution = None

  # these args get passed to your train.py
  # MODIFY AS NEEDED
  hyperparameters = {
    'model_name': model_name,
    'steps': steps,
    'batch_size_gpl': batch_size_gpl,
    'batch_size_qgen': batch_size_qgen
  }

  # MODIFY AS NEEDED - THIS IS FOR TENSORFLOW ONLY
  # CHECK PYTHON SAGEMAKER SDK FOR OTHER FRAMEWORKS (PyTorch, Sklearn, etc...)
  estimator = HuggingFace(
    sagemaker_session=sess,
    role=os.getenv('SM_EXECUTION_ARN'),
    model_dir=model_dir,
    instance_type=instance_type,
    instance_count=instance_count,
    distribution=distribution,
    volume_size=volume_size,
    hyperparameters=hyperparameters,
    metric_definitions=metric_def_binary,
    output_path=output_path,
    source_dir='src',
    entry_point='train.py',
    base_job_name=job_name,
    py_version='py36',
    transformers_version='4.6.1',
    pytorch_version='1.7.1'
  )

  return estimator


def train(estimator, train):
  distribution = 'FullyReplicated'
  
  train_data = sagemaker.inputs.TrainingInput(
    train,
    distribution=distribution,
  )

  inputs = {
    'train': train_data
  }

  estimator.fit(inputs)


if __name__ == "__main__":
  args, _ = parse_args()
  print(args)
  
  # MODIFY AS NEEDED TO MATCH YOUR get_estimator() ABOVE
  estimator = get_estimator(
    args.job_name,
    args.instance_type,
    args.instance_count,
    args.profile_name,
    args.volume_size,
    args.output_path,
    args.model_dir,
    args.model_name,
    args.steps,
    args.batch_size_gpl,
    args.batch_size_qgen
  )

  train(estimator, args.train)