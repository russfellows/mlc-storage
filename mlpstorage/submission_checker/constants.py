from .parsers.json_parser import JSONParser
from .parsers.yaml_parser import YamlParser

VERSIONS = ["v2.0", "v3.0"]
VALID_DIVISIONS = ["open", "closed"]

SYSTEM_PATH = {
    "v2.0": "{division}/{submitter}/systems/{system}.yaml",
    "v3.0": "{division}/{submitter}/systems/{system}.yaml",
    "default": "{division}/{submitter}/systems/{system}.yaml",
}

PARSER_MAP = {
    "System": YamlParser,
    "Summary": JSONParser,
    "Metadata": JSONParser,
    "default": JSONParser
}

DATAGEN_REQUIRED_FILES = {
    "v2.0": ["training_datagen.stdout.log", "training_datagen.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
    "v3.0": ["training_datagen.stdout.log", "training_datagen.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
    "default": ["training_datagen.stdout.log", "training_datagen.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
}

DATAGEN_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}

RUN_REQUIRED_FILES = {
    "v2.0": ["training_run.stdout.log", "training_run.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
    "v3.0": ["training_run.stdout.log", "training_run.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
    "default": ["training_run.stdout.log", "training_datagen.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
}

RUN_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}

CHECKPOINT_REQUIRED_FILES = {
    "v2.0": ["training_run.stdout.log", "training_run.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
    "v3.0": ["training_run.stdout.log", "training_run.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
    "default": ["training_run.stdout.log", "training_datagen.stderr.log", "*output.json", "*per_epoch_stats.json", "*summary.json", "dlio.log"],
}

CHECKPOINT_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}