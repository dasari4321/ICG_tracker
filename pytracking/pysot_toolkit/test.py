
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
#from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES']='2'
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
#import os

import cv2
import torch
import numpy as np
import importlib

from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory
from pysot_toolkit.toolkit.utils.region import vot_overlap, vot_float2str
#from pysot_toolkit.trackers.tracker import Tracker
#from pytracking.evaluation import Tracker
#from pysot_toolkit.trackers.net_wrappers import NetWithBackbone

parser = argparse.ArgumentParser(description='transt tracking')
parser.add_argument('--dataset', '-d', default='LaSOT',type=str,
        help='datasets')
parser.add_argument('--video', default='', type=str,
        help='eval one special video') #Bird1 #Board
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
parser.add_argument('--name', default='eco', type=str,
        help='name of results')
parser.add_argument('--tname', default='eco', type=str, help='Name of tracking method.')
parser.add_argument('--tparam', type=str, default='res', help='Name of parameter file.')
#edg edg_res hog hog_res res res_edg_hog edg_ft
args = parser.parse_args()

torch.set_num_threads(1)
#writer = SummaryWriter(log_dir='/media/hd2/mohana/pytracking/pysot_toolkit/logs/')
def main():
    # load config

    dataset_root = '/media/hd2/mohana/Operation_RedGiant/ALTO/DATASETS/'+args.dataset #Absolute path of the dataset

####################    # create model
    param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(args.tname, args.tparam))
    params = param_module.parameters()
    tracker_module = importlib.import_module('pytracking.tracker.{}'.format(args.tname))
    tracker_class = tracker_module.get_tracker_class()
    args.name = args.name +'_'+ args.tparam
####################
    tracker = tracker_class(params)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.name
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            scores = []
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-w/2, cy-h/2, w, h]
                    init_info = {'init_bbox':gt_bbox_}
                    tracker.initialize(img, init_info)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    info = {}
                    outputs = tracker.track(img, info)
                    pred_bbox = outputs['target_bbox']
                    # if cfg.MASK.MASK:
                    #     pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    scores.append(overlap)
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
#                    writer.add_scalar(scalar_value=pred_bbox,tag='bbox')
#                    writer.add_image('input',torch.from_numpy(img).permute(2,1,0))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
#                    if cv2.waitKey() & 0xFF == ord('q'):
#                        break
            cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('./results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}  mIOU:{:3.2f}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number, np.array(scores).mean()))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-w/2, cy-h/2, w, h]
                    init_info = {'init_bbox':gt_bbox_}
                    tracker.initialize(img, init_info)
                    pred_bbox = gt_bbox_
#                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['target_bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0])))
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps mIOU:{:3.2f}'.format(
                v_idx+1, video.name, toc, idx / toc, np.array(scores).mean()))

#    writer.flush()
#    writer.close()

if __name__ == '__main__':
    main()
