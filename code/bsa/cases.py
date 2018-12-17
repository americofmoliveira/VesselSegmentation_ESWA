import os

from bsa.DRIVE_files import DRIVEFiles
from bsa.STARE_files import STAREFiles
from bsa.CHASE_files import CHASEFiles


#
# ===== ------------------------------
def list_cases(dataset, fold=None):
    """
    Export the codes of the images to be predicted, given the database and the fold.

    Parameters
    ----------
    dataset: python string.
        Dataset name ('DRIVE', 'STARE', or 'CHASE_DB1').
    fold: int
        Number of the fold to be predicted. It must be different from None for STARE and
        CHASE_DB1 datasets.

    Returns
    -------
    cases: python set.

    """

    cases = set()
    base_dir = os.path.join(os.getcwd(), 'datasets', dataset)

    if dataset == 'DRIVE':
        cases = {case for case in range(1, 21)}
        db = DRIVEFiles(base_dir=base_dir, dataset='Test')

    elif dataset == 'STARE':
        db = STAREFiles(base_dir=base_dir, dataset='Training')

        if fold == 1:
            cases = {'255', '239', '324', '291'}

        elif fold == 2:
            cases = {'162', '163', '004', '005'}

        elif fold == 3:
            cases = {'081', '082', '002', '003'}

        elif fold == 4:
            cases = {'077', '240', '001', '319'}

        elif fold == 5:
            cases = {'235', '236', '044', '139'}

    elif dataset == 'CHASE':
        db = CHASEFiles(base_dir=base_dir, dataset='Training')
        if fold == 1:
            cases = {'01R', '01L', '02R', '02L', '03R', '03L', '04R'}

        elif fold == 2:
            cases = {'04L', '05R', '05L', '06R', '06L', '07R', '07L'}

        elif fold == 3:
            cases = {'08R', '08L', '09R', '09L', '10R', '10L', '11R'}

        elif fold == 4:
            cases = {'11L', '12R', '12L', '13R', '13L', '14R', '14L'}

    return cases, db
