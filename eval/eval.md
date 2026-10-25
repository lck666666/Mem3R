# Evaluation data and entry points

See the [README](../README.md) for model installation and benchmark commands.

## Dataset preparation

Evaluation expects preprocessed images and synchronized ground truth in the
layouts below. The original long-sequence preprocessing scripts are not bundled.
Use the original benchmark splits for comparisons with the paper.

Dataset downloads and upstream preparation are described by
[MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md)
and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md).
Check the dataset providers' terms and retain their citations.

The current metadata and loaders expect:

| Benchmark | Prepared input layout |
|---|---|
| TUM long pose | `data/long_tum_s1/<sequence>/rgb_<N>/*.png`, synchronized `groundtruth_<N>.txt` |
| ScanNet long pose | `data/long_scannet_s3/<sequence>/color_<N>/*`, synchronized `pose_<N>.txt` |
| Bonn long depth | `data/long_bonn_s1/rgbd_bonn_dataset/rgbd_bonn_<sequence>/{rgb_<N>,depth_<N>}/*.png` |
| KITTI long depth | `data/long_kitti_s1/depth_selection/val_selection_cropped/{image_gathered_<N>,groundtruth_depth_gathered_<N>}/<sequence>/*.png` |
| 7-Scenes | `<root>/<scene>/TestSplit.txt`, `<scene>/seq-XX/frame-XXXXXX.{color.png,depth.proj.png,pose.txt}` |
| NRGBD | Consult `eval/mv_recon/data.py` for scene structure, intrinsics, depth, and pose conventions |

`--img_root` overrides pose/depth inference metadata roots. Bonn scoring also
accepts `--gt_root` and `--seq_list`. For reconstruction, `--datasets` selects only
the required datasets, with `--seven_scenes_root` and `--nrgbd_root` setting roots.
7-Scenes requires registered `depth.proj.png` files; raw depth files cannot simply
be renamed to meet that requirement.

## Shell launchers

Run from the repository root. All launchers use one process and the released
64-view checkpoint by default. For example:

```bash
CUDA_VISIBLE_DEVICES=0 STATE_GATE=channel STATE_UPDATE=ttt3r \
  bash eval/relpose/run_tum.sh
CUDA_VISIBLE_DEVICES=0 STATE_GATE=channel STATE_UPDATE=direct \
  bash eval/video_depth/run_bonn.sh
```

`MODEL_WEIGHTS`, `POSE_MEMORY`, `STATE_GATE`, `STATE_UPDATE`, and `NUM_PROCESSES` override these defaults.
Dataset/sequence-length selections are near the top of each script. Use direct
launcher commands from the README when specifying custom roots or sequences.
Depth scripts compute all three alignment choices: `metric`, `scale`, and
`scale&shift`. Results for different model strategies use separate directories.
