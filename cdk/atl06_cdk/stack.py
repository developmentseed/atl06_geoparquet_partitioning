"""Lambda + Step Functions pipeline.

Inventory → Sample → PlanSplits → Convert → Compact → Finalize.
"""

from pathlib import Path

from aws_cdk import Duration, RemovalPolicy, Size, Stack
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

from atl06_cdk.settings import StackSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_EXCLUDE = [".venv", ".git", ".superpowers", "cdk.out", "cdk/cdk.out", ".env", "**/__pycache__", "tests", "docs"]


class Atl06StacHashStack(Stack):
    def __init__(self, scope: Construct, id: str, *, settings: StackSettings, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        # Must be set before any Distributed Map is created; enables ResultWriterV2 rendering.
        self.node.set_context("@aws-cdk/aws-stepfunctions:useDistributedMapResultWriterV2", True)
        self.settings = settings

        if settings.bucket_name:
            bucket = s3.Bucket.from_bucket_name(self, "Bucket", settings.bucket_name)
        else:
            bucket = s3.Bucket(self, "Bucket", removal_policy=RemovalPolicy.RETAIN)
        secret = secretsmanager.Secret.from_secret_name_v2(
            self, "EarthdataSecret", settings.edl_secret_name
        )
        environment = {
            "ATL06_BUCKET": bucket.bucket_name,
            "ATL06_EDL_SECRET_NAME": settings.edl_secret_name,
            "ATL06_YEAR": str(settings.year),
            "ATL06_TARGET_ROWS_PER_FILE": str(settings.target_rows_per_file),
            "ATL06_ROW_GROUP_SIZE": str(settings.row_group_size),
            "ATL06_SAMPLE_GRANULE_FRACTION": str(settings.sample_granule_fraction),
            "ATL06_SAMPLE_ROW_FRACTION": str(settings.sample_row_fraction),
        }

        def function(name: str, handler: str, memory_mb: int, storage_mb: int = 512):
            fn = lambda_.DockerImageFunction(
                self,
                name,
                code=lambda_.DockerImageCode.from_image_asset(
                    str(REPO_ROOT),
                    file="processor/Dockerfile",
                    cmd=[f"atl06_processor.handlers.{handler}"],
                    platform=ecr_assets.Platform.LINUX_ARM64,
                    exclude=ASSET_EXCLUDE,
                ),
                architecture=lambda_.Architecture.ARM_64,
                memory_size=memory_mb,
                ephemeral_storage_size=Size.mebibytes(storage_mb),
                timeout=Duration.minutes(15),
                environment=environment,
                log_group=logs.LogGroup(
                    self, f"{name}Logs", retention=logs.RetentionDays.ONE_MONTH
                ),
            )
            bucket.grant_read_write(fn)
            secret.grant_read(fn)
            return fn

        inventory = function("Inventory", "inventory_handler", 1024)
        sample = function("Sample", "sample_handler", settings.convert_memory_mb)
        plan_splits = function("PlanSplits", "plan_splits_handler", 4096)
        convert = function("Convert", "convert_handler", settings.convert_memory_mb)
        compact = function(
            "Compact", "compact_handler", settings.compact_memory_mb, storage_mb=10240
        )
        finalize = function("Finalize", "finalize_handler", 2048)

        def invoke(id: str, fn, result_path: str = sfn.JsonPath.DISCARD, selector=None):
            task = tasks.LambdaInvoke(
                self, id, lambda_function=fn, result_path=result_path, result_selector=selector,
            )
            return self._retry(task)

        def distributed_map(id: str, fn, items_key: str, batch: int, concurrency: int):
            dmap = sfn.DistributedMap(
                self,
                id,
                item_reader=sfn.S3JsonItemReader(
                    bucket=bucket, key=sfn.JsonPath.string_at(items_key)
                ),
                item_batcher=sfn.ItemBatcher(
                    max_items_per_batch=batch,
                    batch_input={"run_prefix.$": "$.run_prefix"},
                ),
                max_concurrency=concurrency,
                tolerated_failure_percentage=0,
                result_writer_v2=sfn.ResultWriterV2(bucket=bucket, prefix=f"results/{id}"),
                result_path=sfn.JsonPath.DISCARD,
            )
            dmap.item_processor(
                self._retry(
                    tasks.LambdaInvoke(
                        self, f"{id}Batch", lambda_function=fn, payload_response_only=True
                    )
                )
            )
            return dmap

        definition = (
            invoke(
                "InventoryStep", inventory, "$.inventory",
                {
                    "sample_key.$": "$.Payload.sample_key",
                    "granules_key.$": "$.Payload.granules_key",
                    "granules.$": "$.Payload.granules",
                },
            )
            .next(distributed_map("SampleMap", sample, "$.inventory.sample_key", 5,
                                  settings.convert_concurrency))
            .next(invoke("PlanSplitsStep", plan_splits, "$.plan",
                         {"ranges_key.$": "$.Payload.ranges_key", "ranges.$": "$.Payload.ranges"}))
            .next(distributed_map("ConvertMap", convert, "$.inventory.granules_key",
                                  settings.convert_batch_size, settings.convert_concurrency))
            .next(distributed_map("CompactMap", compact, "$.plan.ranges_key", 1,
                                  settings.compact_concurrency))
            .next(invoke("FinalizeStep", finalize, "$.report", {"report.$": "$.Payload"}))
        )
        self.state_machine = sfn.StateMachine(
            self,
            "Pipeline",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.hours(24),
        )
        bucket.grant_read_write(self.state_machine)

    @staticmethod
    def _retry(task: tasks.LambdaInvoke) -> tasks.LambdaInvoke:
        task.add_retry(
            errors=["Lambda.TooManyRequestsException", "States.TaskFailed"],
            max_attempts=3,
            backoff_rate=2.0,
            interval=Duration.seconds(5),
        )
        return task
