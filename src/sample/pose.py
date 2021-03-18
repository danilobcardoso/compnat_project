
class Pose:
    def __init__(self, skeleton_model):
        self.data = {}
        self.skeleton_model = skeleton_model

    def add_data(self, data_type, data):
        self.data[data_type] = data