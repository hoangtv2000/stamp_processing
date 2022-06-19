"""
    isort: skip_file
"""
import sys

from pkg_resources import DistributionNotFound, get_distribution

__version__ = None
try:
    __version__ = get_distribution("table_reconstruction").version
except DistributionNotFound:
    __version__ == "0.0.0"  # package is not installed
    pass

from stamp_processing.remover import StampRemover
