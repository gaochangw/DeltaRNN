__author__ = "Chang Gao"
__copyright__ = "Copyright 2020"
__credits__ = ["Chang Gao", "Stefan Braun"]
__license__ = "Private"
__version__ = "0.1.0"
__maintainer__ = "Chang Gao"
__email__ = "chang.gao@uzh.ch"
__status__ = "Prototype"

from project import Project
from steps import prepare, feature, train

if __name__ == '__main__':
    proj = Project()

    # Step 0 - Prepare Dataset
    if proj.step == 'prepare':
        print("####################################################################################################\n"
              "# Step 0: Prepare Dataset                                                                                  \n"
              "####################################################################################################")
        prepare.main(proj)
        proj.step_in()

    # Step 1 - Feature Extraction
    if proj.step == 'feature':
        print(
            "####################################################################################################\n"
            "# Step 1: Feature Extraction                                                                                  \n"
            "####################################################################################################")
        feature.main(proj)
        proj.step_in()

    # Step 2 - Pretrain on GRU
    if proj.step == 'pretrain':
        print("####################################################################################################\n"
              "# Step 2: Pretrain                                                                                  \n"
              "####################################################################################################")
        train.main(proj)
        proj.step_in()

    # Step 3 - Retrain
    if proj.step == 'retrain':
        print("####################################################################################################\n"
              "# Step 3: Retrain                                                                                   \n"
              "####################################################################################################")
        train.main(proj)
        proj.step_in()
