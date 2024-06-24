import nibabel as nib
import time
import glob
import torch
import cc3d

from panoptica import Panoptic_Evaluator
from panoptica import UnmatchedInstancePair
from panoptica import NaiveThresholdMatching
from panoptica import MaximizeMergeMatching
from rich import print as pprint

home = '/home/soumya/mets/'
labels = home + '/labels/test'
preds = home + '/preds/test'

label_files = glob.glob(labels + '/*.nii.gz')
pred_files = glob.glob(preds + '/*.nii.gz')
label_files.sort()
pred_files.sort()
label_files = label_files[2:5]
pred_files = pred_files[2:5]

print('Number of files:', len(label_files))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from brats import lesion_dice
from soumya import lesion_dice_gpu
from brats_optimised import lesion_dice_gpu as lesion_dice_gpu_optimised

total_time = []
total_time_gpu = []
opt = []
panoptic_naive = []
panoptic_many = []

for i in range(len(label_files)):
    label = nib.load(label_files[i]).get_fdata()
    pred = nib.load(pred_files[i]).get_fdata()
    print(label_files[i])
    print(pred_files[i])

    label_data = torch.tensor(label).to(device)
    pred_data = torch.tensor(pred).to(device)

    # CPU
    start_time = time.time()
    # print(lesion_dice(label_data, pred_data))
    score_cpu = lesion_dice(label_data, pred_data)
    end_time = time.time()
    total_time.append(end_time - start_time)

    # GPU
    start_time_gpu = time.time()
    # print(lesion_dice_gpu(pred_data, label_data))
    score_gpu = lesion_dice_gpu(pred_data, label_data)
    end_time_gpu = time.time()
    total_time_gpu.append(end_time_gpu - start_time_gpu)

    # CPU optimised
    start_time_opt = time.time()
    # print(lesion_dice_gpu_optimised(pred_data, label_data))
    score_c = lesion_dice_gpu_optimised(pred_data, label_data)
    end_time_opt = time.time()
    opt.append(end_time_opt - start_time_opt)

    gt = label_data.cpu().numpy()
    pred = pred_data.cpu().numpy()
    pred = cc3d.connected_components(pred, connectivity=26)
    gt = cc3d.connected_components(gt, connectivity=26)

    # Panoptica Naive
    start_time_panoptica = time.time()
    sample = UnmatchedInstancePair(pred, gt)
    evaluator = Panoptic_Evaluator(
        expected_input=UnmatchedInstancePair,
        instance_matcher=NaiveThresholdMatching(),
    )
    result, _ = evaluator.evaluate(sample)
    end_time_panoptica = time.time()
    panoptic_naive.append(end_time_panoptica - start_time_panoptica)

    # Panoptica Many
    start_time_panoptica_many = time.time()
    evaluator_many = Panoptic_Evaluator(
        expected_input=UnmatchedInstancePair,
        instance_matcher=MaximizeMergeMatching(),
    )
    result_many, _ = evaluator_many.evaluate(sample)
    end_time_panoptica_many = time.time()
    panoptic_many.append(end_time_panoptica_many - start_time_panoptica_many)

    print('Time taken (CPU):', total_time[-1])
    print('Time taken (GPU):', total_time_gpu[-1])
    print('Time taken (CPU optimised):', opt[-1])
    print('Time taken (Naive):', panoptic_naive[-1])
    print('Time taken (Many):', panoptic_many[-1])

    print('Score (CPU):', score_cpu)
    print('Score (GPU):', score_gpu)
    print('Score (CPU optimised):', score_c)
    pprint(f"{result.pq_dsc=}")
    pprint(f"{result_many.pq_dsc=}")

    print()

print('Average time taken (CPU):', sum(total_time) / len(total_time))
print('Average time taken (GPU):', sum(total_time_gpu) / len(total_time_gpu))
print('Average time taken (CPU optimised):', sum(opt) / len(opt))
print('Average time taken (Naive):', sum(panoptic_naive) / len(panoptic_naive))
print('Average time taken (Many):', sum(panoptic_many) / len(panoptic_many))