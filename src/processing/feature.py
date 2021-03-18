
class ScaleNormalization(object):

    def __call__(self, sample):
        min_y = float("inf")
        max_y = -float("inf")
        min_x = float("inf")
        max_x = -float("inf")
        min_z = float("inf")
        max_z = -float("inf")
        for skeleton in sample.poses:
            min_x = min(min_x, skeleton.data['position_xyz'][:, 0].min())
            max_x = max(max_x, skeleton.data['position_xyz'][:, 0].max())
            min_y = min(min_y, skeleton.data['position_xyz'][:, 1].min())
            max_y = max(max_y, skeleton.data['position_xyz'][:, 1].max())
            min_z = min(min_z, skeleton.data['position_xyz'][:, 2].min())
            max_z = max(max_z, skeleton.data['position_xyz'][:, 2].max())

        scale = (max_y - min_y) / 2  # A mesma escala será usada nas 3 dimensões para preservar proporção

        move_x = -(min_x + max_x)/2
        move_y = -(min_y + max_y)/2
        move_z = -(min_z + max_z)/2

        for skeleton in sample.poses:
            skeleton.data['position_xyz'][:, 0] = (skeleton.data['position_xyz'][:, 0] + move_x) / scale
            skeleton.data['position_xyz'][:, 1] = (skeleton.data['position_xyz'][:, 1] + move_y) / scale
            skeleton.data['position_xyz'][:, 2] = (skeleton.data['position_xyz'][:, 2] + move_z) / scale

