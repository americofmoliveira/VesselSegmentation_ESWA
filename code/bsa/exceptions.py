"""
Generates patches for training and test dataset images for the DRIVE database.
"""
__authors__ = "Carlos A. Silva"
__copyright__ = "Copyright 2013-2015, University of Minho"
__credits__ = ["Carlos A. Silva"]
__license__ = "Any use is forbidden"
__maintainer__ = "Carlos A. Silva"


#
# ===== ------------------------------
class BSAError(Exception):
    pass


#
# ===== ------------------------------
class StopPatchIteration(BSAError):
    '''
    This exception signals that we have exhausted the patches for the current
    epoch.
    '''
    pass


#
# ===== ------------------------------
class StopContainerIteration(BSAError):
    '''
    This exception indicates to the higher hierarchy that we have finished
    the count of minibatch containers.
    '''
    pass
