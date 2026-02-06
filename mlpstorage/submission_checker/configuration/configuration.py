from ..constants import *

class Config:
    def __init__(self, version, submitters):
        self.version = version
        self.submitters = submitters
        
    def check_submitter(self, submitter):
        if self.submitters is None:
            return True
        return submitter in self.submitters
    
    def get_datagen_required_files(self):
        return DATAGEN_REQUIRED_FILES[self.version]
    
    def get_run_required_files(self):
        return RUN_REQUIRED_FILES[self.version]
    
    def get_checkpoint_required_files(self):
        return CHECKPOINT_REQUIRED_FILES[self.version]
    
    def get_datagen_required_folders(self):
        return DATAGEN_REQUIRED_FOLDERS[self.version]
    
    def get_run_required_folders(self):
        return RUN_REQUIRED_FOLDERS[self.version]
    
    def get_checkpoint_required_folders(self):
        return CHECKPOINT_REQUIRED_FOLDERS[self.version]