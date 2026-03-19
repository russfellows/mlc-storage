# VERSION
VERSION = "2.0.0b1"

# boto3/botocore are banned — install the blocker immediately so any
# transitive import attempt is caught regardless of which module triggers it.
from mlpstorage.ban_boto3 import install as _ban_boto3
_ban_boto3()
__version__ = VERSION