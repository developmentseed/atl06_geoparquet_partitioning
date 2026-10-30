import json
import os

import pytest

os.environ.setdefault("JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION", "1")

from atl06_cdk.settings import StackSettings  # noqa: E402
from atl06_cdk.stack import Atl06StacHashStack  # noqa: E402
from aws_cdk import App, Environment  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402


@pytest.fixture(scope="module")
def template():
    settings = StackSettings(account="123456789012")
    app = App()
    stack = Atl06StacHashStack(
        app, "Atl06", settings=settings,
        env=Environment(account=settings.account, region=settings.region),
    )
    return Template.from_stack(stack)


def definition(template) -> str:
    sm = next(iter(template.find_resources("AWS::StepFunctions::StateMachine").values()))
    return json.dumps(sm["Properties"]["DefinitionString"])


def test_seven_image_lambdas(template):
    fns = template.find_resources("AWS::Lambda::Function", {"Properties": {"PackageType": "Image"}})
    # inventory, sample, plan_splits, convert, prepare_compact, compact, finalize
    assert len(fns) == 7


def test_from_step_choice_routes_to_prepare_compact(template):
    d = definition(template)
    assert '\\"Type\\":\\"Choice\\"' in d and '\\"Variable\\":\\"$.from_step\\"' in d
    assert '\\"StringEquals\\":\\"compact\\"' in d
    assert '\\"ResultPath\\":\\"$.plan\\"' in d
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {"ImageConfig": {"Command": ["atl06_processor.handlers.prepare_compact_handler"]}},
    )


def test_max_rows_per_file_passed_to_lambdas(template):
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {"Environment": {"Variables": {"ATL06_MAX_ROWS_PER_FILE": "25000000"}}},
    )


def test_state_machine_has_three_distributed_maps(template):
    assert definition(template).count('\\"Mode\\":\\"DISTRIBUTED\\"') == 3


def test_convert_concurrency_from_settings(template):
    assert '\\"MaxConcurrency\\":50' in definition(template)


def test_map_items_read_from_inventory_keys(template):
    d = definition(template)
    for key in ("$.inventory.sample_key", "$.inventory.granules_key", "$.plan.ranges_key"):
        assert key in d


def test_compact_has_10gb_memory_and_storage(template):
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "MemorySize": 10240,
            "EphemeralStorage": {"Size": 10240},
            "ImageConfig": {"Command": ["atl06_processor.handlers.compact_handler"]},
        },
    )


def test_functions_are_arm64_with_15_min_timeout(template):
    fns = template.find_resources("AWS::Lambda::Function", {"Properties": {"PackageType": "Image"}})
    for fn in fns.values():
        assert fn["Properties"]["Architectures"] == ["arm64"]
        assert fn["Properties"]["Timeout"] == 900


def test_batch_input_resolves_run_prefix(template):
    assert '\\"BatchInput\\":{\\"run_prefix.$\\":\\"$.run_prefix\\"}' in definition(template)


def test_maps_write_results_to_s3(template):
    assert definition(template).count('\\"ResultWriter\\"') == 3


def test_state_machine_timeout_is_seven_days(template):
    assert '\\"TimeoutSeconds\\":604800' in definition(template)


def test_inventory_has_3gb_for_a_full_year_of_cmr_results(template):
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "MemorySize": 3008,
            "ImageConfig": {"Command": ["atl06_processor.handlers.inventory_handler"]},
        },
    )
