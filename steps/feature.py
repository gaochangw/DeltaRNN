# __author__ = "Chang Gao"
# __copyright__ = "Copyright 2018 to the author"
# __license__ = "Private"
# __version__ = "0.1.0"
# __maintainer__ = "Chang Gao"
# __email__ = "chang.gao@uzh.ch"    `
# __status__ = "Prototype"
import os
import errno
from modules.gscdv2.feat_extract import FeatExtractor


def main(proj):
    # Create feature folder
    try:
        os.makedirs('feat/' + proj.dataset)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Extract Features
    extractor = FeatExtractor(proj)
    extractor.extract(proj)

