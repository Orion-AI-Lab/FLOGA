## FLOGA and BAM-CD

This repository contains the dataset and code for the paper **"FLOGA: A machine learning ready dataset, a benchmark and a novel deep learning model for burnt area mapping with Sentinel-2"** (Sdraka et al., 2023).

## FLOGA Dataset

![FLOGA sample](assets/dataset_sample.png)

You can download the FLOGA dataset from [Dropbox](https://www.dropbox.com/scl/fo/3sqbs3tioox7s5vb4jmwl/h?rlkey=5p3e7wa5al4cy9x34pmtp9g6d&dl=0).

In order to read the downloaded .hdf files, the `hdf5plugin`  python module is needed because data have been compressed using BZip2. These files include the raw Sentinel-2 and MODIS imagery aligned to a common grid, along with the labels and the various masks.

#### Set up
After downloading the .hdf files, you can create an analysis-ready dataset with the `create_dataset.py` script. This python script reads the .hdf files, crops the images into smaller patches and then performs a train/val/test split on the patches.

For example,

```
python create_dataset.py --floga_path path/to/hdf/files --out_path data/ --out_size 256 256 --sample 1
```

The above command will crop the images into 256x256 patches and export 3 pickle files with the train, validation and test splits respectively. The option `--sample` dictates that for each positive patch (i.e. patch that contains at least 1 burnt pixel) a negative one (i.e. a patch with no burnt pixels) will be included. Run `python create_dataset.py --help` for more information on the various options.

#### Dataset exploration
A useful notebook with an exploration of the dataset can be found in `Data_exploration.ipynb`.

#### Benchmark data splits
The train/val/test splits used in the paper can be found [here](https://www.dropbox.com/scl/fi/vq3tl8w5ex23lt1k7z89e/data_split.csv?rlkey=v3ph1xvfykhiljkg6rzlsytq2&dl=0). A ratio of 1:1 was selected (1 negative patch for each positive patch) and sea and cloud patches were removed.

## BAM-CD

![BAM-CD architecture](assets/bam-cd.png)

This repo also contains the code for the proposed BAM-CD model for burnt area mapping with bitemporal Sentinel-2 imagery. The model can be found inside the folder `models/bam_cd/`.

#### Pretrained model

You can find the weights of the pretrained BAM-CD model [here](https://www.dropbox.com/scl/fo/9fia0j00h539t9x6gvc9z/h?rlkey=rvl5bsmx1au796x5z76jkmmgb&dl=0).

### Citation
If you would like to use our work, please cite our paper:

```
@article{sdraka2023floga,
  title={FLOGA: A machine learning ready dataset, a benchmark and a novel deep learning model for burnt area mapping with Sentinel-2},
  author={Sdraka, Maria and Dimakos, Alkinoos and Malounis, Alexandros and Ntasiou, Zisoula and Karantzalos, Konstantinos and Michail, Dimitrios and Papoutsis, Ioannis},
  journal={arXiv preprint arXiv:2311.03339},
  year={2023}
}
```
