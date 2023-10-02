import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class NFSDataset(BaseDataset):
    """ NFS dataset.

    Publication:
        Need for Speed: A Benchmark for Higher Frame Rate Object Tracking
        H. Kiani Galoogahi, A. Fagg, C. Huang, D. Ramanan, and S.Lucey
        ICCV, 2017
        http://openaccess.thecvf.com/content_ICCV_2017/papers/Galoogahi_Need_for_Speed_ICCV_2017_paper.pdf

    Download the dataset from http://ci2cv.net/nfs/index.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.nfs_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float64,backend = 'pandas')

        return Sequence(sequence_info['name'], frames, 'nfs', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "nfs_Gymnastics", "path": "240/Gymnastics", "startFrame": 1, "endFrame": 368, "nz": 5, "ext": "jpg", "anno_path": "240/Gymnastics.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_MachLoop_jet", "path": "240/MachLoop_jet", "startFrame": 1, "endFrame": 99, "nz": 5, "ext": "jpg", "anno_path": "240/MachLoop_jet.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "nfs_Skiing_red", "path": "240/Skiing_red", "startFrame": 1, "endFrame": 69, "nz": 5, "ext": "jpg", "anno_path": "240/Skiing_red.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_Skydiving", "path": "240/Skydiving", "startFrame": 1, "endFrame": 196, "nz": 5, "ext": "jpg", "anno_path": "240/Skydiving.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_airboard_1", "path": "240/airboard_1", "startFrame": 1, "endFrame": 425, "nz": 5, "ext": "jpg", "anno_path": "240/airboard_1.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_airtable_3", "path": "240/airtable_3", "startFrame": 1, "endFrame": 482, "nz": 5, "ext": "jpg", "anno_path": "240/airtable_3.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_1", "path": "240/basketball_1", "startFrame": 1, "endFrame": 282, "nz": 5, "ext": "jpg", "anno_path": "240/basketball_1.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_2", "path": "240/basketball_2", "startFrame": 1, "endFrame": 102, "nz": 5, "ext": "jpg", "anno_path": "240/basketball_2.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_3", "path": "240/basketball_3", "startFrame": 1, "endFrame": 421, "nz": 5, "ext": "jpg", "anno_path": "240/basketball_3.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_6", "path": "240/basketball_6", "startFrame": 1, "endFrame": 224, "nz": 5, "ext": "jpg", "anno_path": "240/basketball_6.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_7", "path": "240/basketball_7", "startFrame": 1, "endFrame": 240, "nz": 5, "ext": "jpg", "anno_path": "240/basketball_7.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_basketball_player", "path": "240/basketball_player", "startFrame": 1, "endFrame": 369, "nz": 5, "ext": "jpg", "anno_path": "240/basketball_player.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_basketball_player_2", "path": "240/basketball_player_2", "startFrame": 1, "endFrame": 437, "nz": 5, "ext": "jpg", "anno_path": "240/basketball_player_2.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_beach_flipback_person", "path": "240/beach_flipback_person", "startFrame": 1, "endFrame": 61, "nz": 5, "ext": "jpg", "anno_path": "240/beach_flipback_person.txt", "object_class": "person head", 'occlusion': False},
            {"name": "nfs_bee", "path": "240/bee", "startFrame": 1, "endFrame": 45, "nz": 5, "ext": "jpg", "anno_path": "240/bee.txt", "object_class": "insect", 'occlusion': False},
            {"name": "nfs_biker_acrobat", "path": "240/biker_acrobat", "startFrame": 1, "endFrame": 128, "nz": 5, "ext": "jpg", "anno_path": "240/biker_acrobat.txt", "object_class": "bicycle", 'occlusion': False},
            {"name": "nfs_biker_all_1", "path": "240/biker_all_1", "startFrame": 1, "endFrame": 113, "nz": 5, "ext": "jpg", "anno_path": "240/biker_all_1.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_biker_head_2", "path": "240/biker_head_2", "startFrame": 1, "endFrame": 132, "nz": 5, "ext": "jpg", "anno_path": "240/biker_head_2.txt", "object_class": "person head", 'occlusion': False},
            {"name": "nfs_biker_head_3", "path": "240/biker_head_3", "startFrame": 1, "endFrame": 254, "nz": 5, "ext": "jpg", "anno_path": "240/biker_head_3.txt", "object_class": "person head", 'occlusion': False},
            {"name": "nfs_biker_upper_body", "path": "240/biker_upper_body", "startFrame": 1, "endFrame": 194, "nz": 5, "ext": "jpg", "anno_path": "240/biker_upper_body.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_biker_whole_body", "path": "240/biker_whole_body", "startFrame": 1, "endFrame": 572, "nz": 5, "ext": "jpg", "anno_path": "240/biker_whole_body.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_billiard_2", "path": "240/billiard_2", "startFrame": 1, "endFrame": 604, "nz": 5, "ext": "jpg", "anno_path": "240/billiard_2.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_billiard_3", "path": "240/billiard_3", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg", "anno_path": "240/billiard_3.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_billiard_6", "path": "240/billiard_6", "startFrame": 1, "endFrame": 771, "nz": 5, "ext": "jpg", "anno_path": "240/billiard_6.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_billiard_7", "path": "240/billiard_7", "startFrame": 1, "endFrame": 724, "nz": 5, "ext": "jpg", "anno_path": "240/billiard_7.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_billiard_8", "path": "240/billiard_8", "startFrame": 1, "endFrame": 778, "nz": 5, "ext": "jpg", "anno_path": "240/billiard_8.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_bird_2", "path": "240/bird_2", "startFrame": 1, "endFrame": 476, "nz": 5, "ext": "jpg", "anno_path": "240/bird_2.txt", "object_class": "bird", 'occlusion': False},
            {"name": "nfs_book", "path": "240/book", "startFrame": 1, "endFrame": 288, "nz": 5, "ext": "jpg", "anno_path": "240/book.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_bottle", "path": "240/bottle", "startFrame": 1, "endFrame": 2103, "nz": 5, "ext": "jpg", "anno_path": "240/bottle.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_bowling_1", "path": "240/bowling_1", "startFrame": 1, "endFrame": 303, "nz": 5, "ext": "jpg", "anno_path": "240/bowling_1.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_bowling_2", "path": "240/bowling_2", "startFrame": 1, "endFrame": 710, "nz": 5, "ext": "jpg", "anno_path": "240/bowling_2.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_bowling_3", "path": "240/bowling_3", "startFrame": 1, "endFrame": 271, "nz": 5, "ext": "jpg", "anno_path": "240/bowling_3.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_bowling_6", "path": "240/bowling_6", "startFrame": 1, "endFrame": 260, "nz": 5, "ext": "jpg", "anno_path": "240/bowling_6.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_bowling_ball", "path": "240/bowling_ball", "startFrame": 1, "endFrame": 275, "nz": 5, "ext": "jpg", "anno_path": "240/bowling_ball.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_bunny", "path": "240/bunny", "startFrame": 1, "endFrame": 705, "nz": 5, "ext": "jpg", "anno_path": "240/bunny.txt", "object_class": "mammal", 'occlusion': False},
            {"name": "nfs_car", "path": "240/car", "startFrame": 1, "endFrame": 2020, "nz": 5, "ext": "jpg", "anno_path": "240/car.txt", "object_class": "car", 'occlusion': True},
            {"name": "nfs_car_camaro", "path": "240/car_camaro", "startFrame": 1, "endFrame": 36, "nz": 5, "ext": "jpg", "anno_path": "240/car_camaro.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_drifting", "path": "240/car_drifting", "startFrame": 1, "endFrame": 173, "nz": 5, "ext": "jpg", "anno_path": "240/car_drifting.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_jumping", "path": "240/car_jumping", "startFrame": 1, "endFrame": 22, "nz": 5, "ext": "jpg", "anno_path": "240/car_jumping.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_rc_rolling", "path": "240/car_rc_rolling", "startFrame": 1, "endFrame": 62, "nz": 5, "ext": "jpg", "anno_path": "240/car_rc_rolling.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_rc_rotating", "path": "240/car_rc_rotating", "startFrame": 1, "endFrame": 80, "nz": 5, "ext": "jpg", "anno_path": "240/car_rc_rotating.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_side", "path": "240/car_side", "startFrame": 1, "endFrame": 108, "nz": 5, "ext": "jpg", "anno_path": "240/car_side.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_white", "path": "240/car_white", "startFrame": 1, "endFrame": 2063, "nz": 5, "ext": "jpg", "anno_path": "240/car_white.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_cheetah", "path": "240/cheetah", "startFrame": 1, "endFrame": 167, "nz": 5, "ext": "jpg", "anno_path": "240/cheetah.txt", "object_class": "mammal", 'occlusion': True},
            {"name": "nfs_cup", "path": "240/cup", "startFrame": 1, "endFrame": 1281, "nz": 5, "ext": "jpg", "anno_path": "240/cup.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_cup_2", "path": "240/cup_2", "startFrame": 1, "endFrame": 182, "nz": 5, "ext": "jpg", "anno_path": "240/cup_2.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_dog", "path": "240/dog", "startFrame": 1, "endFrame": 1030, "nz": 5, "ext": "jpg", "anno_path": "240/dog.txt", "object_class": "dog", 'occlusion': True},
            {"name": "nfs_dog_1", "path": "240/dog_1", "startFrame": 1, "endFrame": 168, "nz": 5, "ext": "jpg", "anno_path": "240/dog_1.txt", "object_class": "dog", 'occlusion': False},
            {"name": "nfs_dog_2", "path": "240/dog_2", "startFrame": 1, "endFrame": 594, "nz": 5, "ext": "jpg", "anno_path": "240/dog_2.txt", "object_class": "dog", 'occlusion': True},
            {"name": "nfs_dog_3", "path": "240/dog_3", "startFrame": 1, "endFrame": 200, "nz": 5, "ext": "jpg", "anno_path": "240/dog_3.txt", "object_class": "dog", 'occlusion': False},
            {"name": "nfs_dogs", "path": "240/dogs", "startFrame": 1, "endFrame": 198, "nz": 5, "ext": "jpg", "anno_path": "240/dogs.txt", "object_class": "dog", 'occlusion': True},
            {"name": "nfs_dollar", "path": "240/dollar", "startFrame": 1, "endFrame": 1426, "nz": 5, "ext": "jpg", "anno_path": "240/dollar.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_drone", "path": "240/drone", "startFrame": 1, "endFrame": 70, "nz": 5, "ext": "jpg", "anno_path": "240/drone.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "nfs_ducks_lake", "path": "240/ducks_lake", "startFrame": 1, "endFrame": 107, "nz": 5, "ext": "jpg", "anno_path": "240/ducks_lake.txt", "object_class": "bird", 'occlusion': False},
            {"name": "nfs_exit", "path": "240/exit", "startFrame": 1, "endFrame": 359, "nz": 5, "ext": "jpg", "anno_path": "240/exit.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_first", "path": "240/first", "startFrame": 1, "endFrame": 435, "nz": 5, "ext": "jpg", "anno_path": "240/first.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_flower", "path": "240/flower", "startFrame": 1, "endFrame": 448, "nz": 5, "ext": "jpg", "anno_path": "240/flower.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_footbal_skill", "path": "240/footbal_skill", "startFrame": 1, "endFrame": 131, "nz": 5, "ext": "jpg", "anno_path": "240/footbal_skill.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_helicopter", "path": "240/helicopter", "startFrame": 1, "endFrame": 310, "nz": 5, "ext": "jpg", "anno_path": "240/helicopter.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "nfs_horse_jumping", "path": "240/horse_jumping", "startFrame": 1, "endFrame": 117, "nz": 5, "ext": "jpg", "anno_path": "240/horse_jumping.txt", "object_class": "horse", 'occlusion': True},
            {"name": "nfs_horse_running", "path": "240/horse_running", "startFrame": 1, "endFrame": 139, "nz": 5, "ext": "jpg", "anno_path": "240/horse_running.txt", "object_class": "horse", 'occlusion': False},
            {"name": "nfs_iceskating_6", "path": "240/iceskating_6", "startFrame": 1, "endFrame": 603, "nz": 5, "ext": "jpg", "anno_path": "240/iceskating_6.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_jellyfish_5", "path": "240/jellyfish_5", "startFrame": 1, "endFrame": 746, "nz": 5, "ext": "jpg", "anno_path": "240/jellyfish_5.txt", "object_class": "invertebrate", 'occlusion': False},
            {"name": "nfs_kid_swing", "path": "240/kid_swing", "startFrame": 1, "endFrame": 169, "nz": 5, "ext": "jpg", "anno_path": "240/kid_swing.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_motorcross", "path": "240/motorcross", "startFrame": 1, "endFrame": 39, "nz": 5, "ext": "jpg", "anno_path": "240/motorcross.txt", "object_class": "vehicle", 'occlusion': True},
            {"name": "nfs_motorcross_kawasaki", "path": "240/motorcross_kawasaki", "startFrame": 1, "endFrame": 65, "nz": 5, "ext": "jpg", "anno_path": "240/motorcross_kawasaki.txt", "object_class": "vehicle", 'occlusion': False},
            {"name": "nfs_parkour", "path": "240/parkour", "startFrame": 1, "endFrame": 58, "nz": 5, "ext": "jpg", "anno_path": "240/parkour.txt", "object_class": "person head", 'occlusion': False},
            {"name": "nfs_person_scooter", "path": "240/person_scooter", "startFrame": 1, "endFrame": 413, "nz": 5, "ext": "jpg", "anno_path": "240/person_scooter.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_pingpong_2", "path": "240/pingpong_2", "startFrame": 1, "endFrame": 1277, "nz": 5, "ext": "jpg", "anno_path": "240/pingpong_2.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_pingpong_7", "path": "240/pingpong_7", "startFrame": 1, "endFrame": 1290, "nz": 5, "ext": "jpg", "anno_path": "240/pingpong_7.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_pingpong_8", "path": "240/pingpong_8", "startFrame": 1, "endFrame": 296, "nz": 5, "ext": "jpg", "anno_path": "240/pingpong_8.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_rubber", "path": "240/rubber", "startFrame": 1, "endFrame": 1328, "nz": 5, "ext": "jpg", "anno_path": "240/rubber.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_running", "path": "240/running", "startFrame": 1, "endFrame": 677, "nz": 5, "ext": "jpg", "anno_path": "240/running.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_running_100_m", "path": "240/running_100_m", "startFrame": 1, "endFrame": 141, "nz": 5, "ext": "jpg", "anno_path": "240/running_100_m.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_running_100_m_2", "path": "240/running_100_m_2", "startFrame": 1, "endFrame": 337, "nz": 5, "ext": "jpg", "anno_path": "240/running_100_m_2.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_running_2", "path": "240/running_2", "startFrame": 1, "endFrame": 237, "nz": 5, "ext": "jpg", "anno_path": "240/running_2.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_shuffleboard_1", "path": "240/shuffleboard_1", "startFrame": 1, "endFrame": 42, "nz": 5, "ext": "jpg", "anno_path": "240/shuffleboard_1.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffleboard_2", "path": "240/shuffleboard_2", "startFrame": 1, "endFrame": 41, "nz": 5, "ext": "jpg", "anno_path": "240/shuffleboard_2.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffleboard_4", "path": "240/shuffleboard_4", "startFrame": 1, "endFrame": 62, "nz": 5, "ext": "jpg", "anno_path": "240/shuffleboard_4.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffleboard_5", "path": "240/shuffleboard_5", "startFrame": 1, "endFrame": 32, "nz": 5, "ext": "jpg", "anno_path": "240/shuffleboard_5.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffleboard_6", "path": "240/shuffleboard_6", "startFrame": 1, "endFrame": 52, "nz": 5, "ext": "jpg", "anno_path": "240/shuffleboard_6.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffletable_2", "path": "240/shuffletable_2", "startFrame": 1, "endFrame": 372, "nz": 5, "ext": "jpg", "anno_path": "240/shuffletable_2.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffletable_3", "path": "240/shuffletable_3", "startFrame": 1, "endFrame": 368, "nz": 5, "ext": "jpg", "anno_path": "240/shuffletable_3.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffletable_4", "path": "240/shuffletable_4", "startFrame": 1, "endFrame": 101, "nz": 5, "ext": "jpg", "anno_path": "240/shuffletable_4.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_ski_long", "path": "240/ski_long", "startFrame": 1, "endFrame": 274, "nz": 5, "ext": "jpg", "anno_path": "240/ski_long.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_soccer_ball", "path": "240/soccer_ball", "startFrame": 1, "endFrame": 163, "nz": 5, "ext": "jpg", "anno_path": "240/soccer_ball.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_soccer_ball_2", "path": "240/soccer_ball_2", "startFrame": 1, "endFrame": 435, "nz": 5, "ext": "jpg", "anno_path": "240/soccer_ball_2.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_soccer_ball_3", "path": "240/soccer_ball_3", "startFrame": 1, "endFrame": 1381, "nz": 5, "ext": "jpg", "anno_path": "240/soccer_ball_3.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_soccer_player_2", "path": "240/soccer_player_2", "startFrame": 1, "endFrame": 475, "nz": 5, "ext": "jpg", "anno_path": "240/soccer_player_2.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_soccer_player_3", "path": "240/soccer_player_3", "startFrame": 1, "endFrame": 319, "nz": 5, "ext": "jpg", "anno_path": "240/soccer_player_3.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_stop_sign", "path": "240/stop_sign", "startFrame": 1, "endFrame": 302, "nz": 5, "ext": "jpg", "anno_path": "240/stop_sign.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_suv", "path": "240/suv", "startFrame": 1, "endFrame": 1444, "nz": 5, "ext": "jpg", "anno_path": "240/suv.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_tiger", "path": "240/tiger", "startFrame": 1, "endFrame": 1556, "nz": 5, "ext": "jpg", "anno_path": "240/tiger.txt", "object_class": "mammal", 'occlusion': False},
            {"name": "nfs_walking", "path": "240/walking", "startFrame": 1, "endFrame": 555, "nz": 5, "ext": "jpg", "anno_path": "240/walking.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_walking_3", "path": "240/walking_3", "startFrame": 1, "endFrame": 1427, "nz": 5, "ext": "jpg", "anno_path": "240/walking_3.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_water_ski_2", "path": "240/water_ski_2", "startFrame": 1, "endFrame": 47, "nz": 5, "ext": "jpg", "anno_path": "240/water_ski_2.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_yoyo", "path": "240/yoyo", "startFrame": 1, "endFrame": 67, "nz": 5, "ext": "jpg", "anno_path": "240/yoyo.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_zebra_fish", "path": "240/zebra_fish", "startFrame": 1, "endFrame": 671, "nz": 5, "ext": "jpg", "anno_path": "240/zebra_fish.txt", "object_class": "fish", 'occlusion': False},
        ]

        return sequence_info_list
