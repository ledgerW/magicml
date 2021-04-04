import os
import json
import boto3

# Load only if in Docker Lambda Environment
if os.getenv('CONTAINER_ENV'):
  import numpy as np
  import pandas as pd
  import tensorflow as tf


def dummy(event, context):
  print(event)
  
  return 'Hello :)'


