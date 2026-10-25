import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.video_depth.tools import depth_evaluation, group_by_directory
import numpy as np
import cv2
from tqdm import tqdm
import glob
from PIL import Image
import argparse
import json
from eval.video_depth.metadata import dataset_metadata


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="directory to save evaluation results",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="",
        help="directory containing inference predictions",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="nyu", choices=list(dataset_metadata.keys())
    )
    parser.add_argument(
        "--align",
        type=str,
        default="scale&shift",
        choices=["scale&shift", "scale", "metric"],
    )
    parser.add_argument("--gt_root", help="Override prepared Bonn ground-truth dataset root")
    parser.add_argument("--seq_list", nargs="+", help="Bonn sequences to score (default: benchmark sequences)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(args):
    if not args.pred_dir:
        args.pred_dir = args.output_dir

        # ADD THIS LINE: Ensure the output directory exists before writing any files
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_dataset == "sintel":
        TAG_FLOAT = 202021.25

        def depth_read(filename):
            """Read depth data from file, return as numpy array."""
            f = open(filename, "rb")
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert (
                check == TAG_FLOAT
            ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
                TAG_FLOAT, check
            )
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            size = width * height
            assert (
                width > 0 and height > 0 and size > 1 and size < 100000000
            ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
                width, height
            )
            depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
            return depth

        pred_pathes = glob.glob(
            f"{args.pred_dir}/*/frame_*.npy"
        )  # Predictions are read from --pred_dir.
        pred_pathes = sorted(pred_pathes)

        if len(pred_pathes) > 643:
            full = True
        else:
            full = False

        if full:
            depth_pathes = glob.glob(f"./data/sintel/training/depth/*/*.dpt")
            depth_pathes = sorted(depth_pathes)
        else:
            seq_list = [
                "alley_2",
                "ambush_4",
                "ambush_5",
                "ambush_6",
                "cave_2",
                "cave_4",
                "market_2",
                "market_5",
                "market_6",
                "shaman_3",
                "sleeping_1",
                "sleeping_2",
                "temple_2",
                "temple_3",
            ]
            depth_pathes_folder = [
                f"./data/sintel/training/depth/{seq}" for seq in seq_list
            ]
            depth_pathes = []
            for depth_pathes_folder_i in depth_pathes_folder:
                depth_pathes += glob.glob(depth_pathes_folder_i + "/*.dpt")
            depth_pathes = sorted(depth_pathes)

        def get_video_results():
            grouped_pred_depth = group_by_directory(pred_pathes)

            grouped_gt_depth = group_by_directory(depth_pathes)
            gathered_depth_metrics = []

            for key in tqdm(grouped_pred_depth.keys()):
                pd_pathes = grouped_pred_depth[key]
                gt_pathes = grouped_gt_depth[key.replace("_pred_depth", "")]

                gt_depth = np.stack(
                    [depth_read(gt_path) for gt_path in gt_pathes], axis=0
                )
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )
                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_lad2=True,
                            use_gpu=args.device == "cuda",
                            post_clip_max=70,
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_scale=True,
                            use_gpu=args.device == "cuda",
                            post_clip_max=70,
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            metric_scale=True,
                            use_gpu=args.device == "cuda",
                            post_clip_max=70,
                        )
                    )
                gathered_depth_metrics.append(depth_results)

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            average_metrics = {
                key: np.average(
                    [metrics[key] for metrics in gathered_depth_metrics],
                    weights=[
                        metrics["valid_pixels"] for metrics in gathered_depth_metrics
                    ],
                )
                for key in gathered_depth_metrics[0].keys()
                if key != "valid_pixels"
            }
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()
    elif args.eval_dataset.startswith("bonn"):

        def depth_read(filename):
            # loads depth map D from png file
            # and returns it as a numpy array
            depth_png = np.asarray(Image.open(filename))
            # make sure we have a proper 16bit depth map here.. not 8bit!
            assert np.max(depth_png) > 255
            depth = depth_png.astype(np.float64) / 5000.0
            depth[depth_png == 0] = -1.0
            return depth

        seq_list = args.seq_list or ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]

        # extract number from dataset name, e.g. bonn_400 -> 400
        if "_" in args.eval_dataset:
            bonn_number = args.eval_dataset.split("_")[-1]
        else:
            bonn_number = "110"  # default value

        gt_root = args.gt_root or ("./data/bonn/rgbd_bonn_dataset" if args.eval_dataset == "bonn" else "./data/long_bonn_s1/rgbd_bonn_dataset")
        img_pathes_folder = [
            f"{gt_root}/rgbd_bonn_{seq}/rgb_{bonn_number}/*.png"
            for seq in seq_list
        ]
        img_pathes = []
        for img_pathes_folder_i in img_pathes_folder:
            img_pathes += glob.glob(img_pathes_folder_i)
        img_pathes = sorted(img_pathes)
        depth_pathes_folder = [
            f"{gt_root}/rgbd_bonn_{seq}/depth_{bonn_number}/*.png"
            for seq in seq_list
        ]
        depth_pathes = []
        for depth_pathes_folder_i in depth_pathes_folder:
            depth_pathes += glob.glob(depth_pathes_folder_i)
        depth_pathes = sorted(depth_pathes)
        pred_pathes = glob.glob(
            f"{args.pred_dir}/*/frame*.npy"
        )  # Predictions are read from --pred_dir.
        pred_pathes = sorted(pred_pathes)

        def get_video_results():
            grouped_pred_depth = group_by_directory(pred_pathes)
            grouped_gt_depth = group_by_directory(depth_pathes, idx=-2)
            gathered_depth_metrics = []
            if not grouped_gt_depth:
                raise ValueError(f"No Bonn ground-truth depth maps found under {gt_root}")
            for key in tqdm(grouped_gt_depth.keys()):
                pd_pathes = grouped_pred_depth[key[10:]]
                gt_pathes = grouped_gt_depth[key]
                if len(pd_pathes) != len(gt_pathes):
                    raise ValueError(f"Prediction/ground-truth count mismatch for {key}: {len(pd_pathes)} vs {len(gt_pathes)}")
                gt_depth = np.stack(
                    [depth_read(gt_path) for gt_path in gt_pathes], axis=0
                )
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )
                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_lad2=True,
                            use_gpu=args.device == "cuda",
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_scale=True,
                            use_gpu=args.device == "cuda",
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            metric_scale=True,
                            use_gpu=args.device == "cuda",
                        )
                    )
                gathered_depth_metrics.append(depth_results)

                # seq_len = gt_depth.shape[0]
                # error_map = error_map.reshape(seq_len, -1, error_map.shape[-1]).cpu()
                # error_map_colored = colorize(error_map, range=(error_map.min(), error_map.max()), append_cbar=True)
                # ImageSequenceClip([x for x in (error_map_colored.numpy()*255).astype(np.uint8)], fps=10).write_videofile(f'{args.output_dir}/errormap_{key}_{args.align}.mp4', fps=10)

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            average_metrics = {
                key: np.average(
                    [metrics[key] for metrics in gathered_depth_metrics],
                    weights=[
                        metrics["valid_pixels"] for metrics in gathered_depth_metrics
                    ],
                )
                for key in gathered_depth_metrics[0].keys()
                if key != "valid_pixels"
            }
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()
    elif args.eval_dataset.startswith("kitti"):

        def depth_read(filename):
            # loads depth map D from png file
            # and returns it as a numpy array,
            # for details see readme.txt
            img_pil = Image.open(filename)
            depth_png = np.array(img_pil, dtype=int)
            # make sure we have a proper 16bit depth map here.. not 8bit!
            assert np.max(depth_png) > 255

            depth = depth_png.astype(float) / 256.0
            depth[depth_png == 0] = -1.0
            return depth

        # extract number from dataset name, e.g. kitti_100 -> 100
        if "_" in args.eval_dataset:
            kitti_number = args.eval_dataset.split("_")[-1]
        else:
            kitti_number = "110"  # default value

        depth_pathes = glob.glob(
            f"./data/long_kitti_s1/depth_selection/val_selection_cropped/groundtruth_depth_gathered_{kitti_number}/*/*.png"
        )
        depth_pathes = sorted(depth_pathes)
        pred_pathes = glob.glob(
            f"{args.pred_dir}/*/frame_*.npy"
        )  # Predictions are read from --pred_dir.
        pred_pathes = sorted(pred_pathes)

        def get_video_results():
            grouped_pred_depth = group_by_directory(pred_pathes)
            grouped_gt_depth = group_by_directory(depth_pathes)
            gathered_depth_metrics = []
            for key in tqdm(grouped_pred_depth.keys()):
                pd_pathes = grouped_pred_depth[key]
                gt_pathes = grouped_gt_depth[key]
                if len(pd_pathes) != len(gt_pathes):
                    raise ValueError(f"Prediction/ground-truth count mismatch for {key}: {len(pd_pathes)} vs {len(gt_pathes)}")
                gt_depth = np.stack(
                    [depth_read(gt_path) for gt_path in gt_pathes], axis=0
                )
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )

                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            align_with_lad2=True,
                            use_gpu=args.device == "cuda",
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            align_with_scale=True,
                            use_gpu=args.device == "cuda",
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            metric_scale=True,
                            use_gpu=args.device == "cuda",
                        )
                    )
                gathered_depth_metrics.append(depth_results)

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            average_metrics = {
                key: np.average(
                    [metrics[key] for metrics in gathered_depth_metrics],
                    weights=[
                        metrics["valid_pixels"] for metrics in gathered_depth_metrics
                    ],
                )
                for key in gathered_depth_metrics[0].keys()
                if key != "valid_pixels"
            }
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
