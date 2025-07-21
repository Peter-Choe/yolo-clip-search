import json
import sys, os
import json
import tempfile
import shutil


def load_metadata(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)
