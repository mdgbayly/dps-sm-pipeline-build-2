"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="DpsPackageGroup",
    pipeline_name="DpsPipeline",
    base_job_prefix="Dps",
    processing_instance_type="ml.t3.medium",
    training_instance_type="ml.m5.large",
):
    """Gets a SageMaker ML Pipeline instance working Distributed Practice Service data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

    # framework_version = "1.0-1"
    framework_version = "0.23-1"

    ###########################
    # Shared? Processor
    pipeline_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-dps-encode",
        role=role,
        sagemaker_session=pipeline_session,
    )

    encode_args = pipeline_processor.run(
        code=os.path.join(BASE_DIR, "encode.py"),
        outputs=[
            ProcessingOutput(
                output_name='train_data',
                source='/opt/ml/processing/train/'
            ),
            ProcessingOutput(
                output_name='test_data',
                source='/opt/ml/processing/test/'
            )
        ],
        arguments=['-i', '-u', '-ic', '-a', '-w', '-tw']
    )
    
    step_encode = ProcessingStep(name="PracticeEncode", step_args=encode_args)
    
    train_args = pipeline_processor.run(
        code=os.path.join(BASE_DIR, "train_eval_lr.py"),
        inputs=[
            ProcessingInput(
                source=step_encode.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                destination='/opt/ml/processing/train'
            ),
            ProcessingInput(
                source=step_encode.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination='/opt/ml/processing/test'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='model_data',
                source='/opt/ml/processing/model/'
            )
        ],
        arguments=['--X_train_file', 'X-train-G-uiicwatw.npz', '--X_test_file', 'X-test-G-uiicwatw.npz', '--user_type', 'G']
    )

    step_train = ProcessingStep(name="PracticeTrain", step_args=train_args)

    update_args = pipeline_processor.run(
        code=os.path.join(BASE_DIR, "update_rpd.py"),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ProcessingOutputConfig.Outputs["model_data"].S3Output.S3Uri,
                destination='/opt/ml/processing/model'
            ),
            ProcessingInput(
                source=step_encode.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination='/opt/ml/processing/test'
            )
        ],
        arguments=['--X_user_practices', 'X-test-G-user-practices.npy', '--y_predictions', 'y-pred-test-G-uiicwatw.npy']
    )

    step_update = ProcessingStep(name="PracticeUpdateRpd", step_args=update_args)

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count
        ],
        steps=[step_encode, step_train, step_update],
        sagemaker_session=pipeline_session,
    )
    return pipeline
