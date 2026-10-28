import aws_cdk as cdk
from atl06_cdk.settings import StackSettings
from atl06_cdk.stack import Atl06StacHashStack

settings = StackSettings()
app = cdk.App()
Atl06StacHashStack(
    app,
    "Atl06StacHash",
    settings=settings,
    env=cdk.Environment(account=settings.account, region=settings.region),
)
app.synth()
