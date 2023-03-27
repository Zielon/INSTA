<h2 align="center"><b>INSTA - Instant Volumetric Head Avatars</b></h2>

<h4 align="center"><b><a href="https://zielon.github.io/" target="_blank">Wojciech Zielonka</a>, <a href="https://sites.google.com/site/bolkartt/" target="_blank">Timo Bolkart</a>, <a href="https://justusthies.github.io/" target="_blank">Justus Thies</a></b></h4>

<h6 align="center"><i>Max Planck Institute for Intelligent Systems, Tübingen, Germany</i></h6>

<h4 align="center">
<a href="https://youtu.be/HOgaeWTih7Q" target="_blank">Video&nbsp</a>
<a href="https://arxiv.org/pdf/2211.12499v2.pdf" target="_blank">Paper&nbsp</a>
<a href="https://zielon.github.io/insta/" target="_blank">Project Website&nbsp</a>
<a href="https://keeper.mpdl.mpg.de/d/5ea4d2c300e9444a8b0b/" target="_blank"><b>Dataset&nbsp</b></a>
<a href="https://github.com/Zielon/metrical-tracker" target="_blank">Face Tracker&nbsp</a>
<a href="https://github.com/Zielon/INSTA-pytorch" target="_blank">INSTA Pytorch&nbsp</a>
<a href="mailto:&#105;&#110;&#115;&#116;&#97;&#64;&#116;&#117;&#101;&#46;&#109;&#112;&#103;&#46;&#100;&#101;">Email</a>
</h4>

<div align="center"> 
<img src="documents/faces.gif">
<br>
<i style="font-size: 1.05em;">Official Repository for CVPR 2023 paper Instant Volumetric Head Avatars</i>
</div>
<br>

This repository is based on [instant-ngp](https://github.com/NVlabs/instant-ngp), some of the features of the original code are not available in this work. Therefore, one should restrain the program options to the main menu only.

<div align="center"> 
&#x26A0 We also prepared a Pytorch demo version of the project <a href="https://github.com/Zielon/INSTA-pytorch" target="_blank">INSTA Pytorch&nbsp</a> &#x26A0
</div>

### Installation

The repository is based on `instant-ngp` [commit](https://github.com/NVlabs/instant-ngp/tree/e7631da9fca9d0f3467f826fccd7a5849b3f6309). The requirements for the installation are the same, therefore please follow the [guide](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux).
Remember to use the `--recursive` option during cloning.

```shell
git clone --recursive https://github.com/Zielon/INSTA.git
cd INSTA
cmake . -B build
cmake --build build --config RelWithDebInfo -j
```

### Usage and Requirements

After building the project you can either start training an avatar from scratch or load a snapshot. For training, we recommend a graphics card higher or equal to `RTX3090 24GB`, (we have not tested any other GPU) and `64 GB` of RAM memory. Rendering from a snapshot does not require a high-end GPU
and can be performed even on a laptop. We have tested it on `RTX 3080 8GB` laptop version.

The viewer options are the same as in the case of [instant-ngp](https://github.com/NVlabs/instant-ngp#keyboard-shortcuts-and-recommended-controls), with some additional key `F` to raycast the FLAME mesh.

Usage example

```shell
# Training
./build/rta --config insta.json --scene data/obama --height 1024 --width 1024

# Loading from a checkpoint
./build/rta --config insta.json --scene data/obama/transforms_test.json --height 1024 --width 1024 --snapshot data/obama/snapshot.msgpack
```

### Dataset and Training

We are releasing part of our dataset together with publicly available preprocessed avatars from [NHA](https://github.com/philgras/neural-head-avatars), [NeRFace](https://github.com/gafniguy/4D-Facial-Avatars) and [IMAvatar](https://github.com/zhengyuf/IMavatar).
The output of the training (**Record Video** in menu), including rendered frames, checkpoint, etc will be saved in the `./data/{actor}/experiments/{config}/debug`.
After the specified number of steps, the program will automatically either render all videos with the `All` option or only the currently selected one in `Mode`.

[Available avatars](https://keeper.mpdl.mpg.de/d/5ea4d2c300e9444a8b0b/). Click the selected avatar to download the training dataset and the checkpoint. The avatars have to be placed in the `data` folder.
<div align="center" dis>
    <table class="images" width="100%"  style="border:0px solid white; width:100%;">
        <tr style="border: 0px;">
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/53e9988994914c93bb51/?dl=1"><img src="documents/gifs/justin.gif" height="128" width="128"></a></td>
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/1a58d09b3b7442988c3e/?dl=1"><img src="documents/gifs/nf_03.gif" height="128" width="128"></a></td>
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/c3632aaba19542d49f1d/?dl=1"><img src="documents/gifs/nf_01.gif" height="128" width="128"></a></td>
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/f273e0d5c6c14d8892a0/?dl=1"><img src="documents/gifs/marcel.gif" height="128" width="128"></a></td>
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/9acb4822310c4d5aa555/?dl=1"><img src="documents/gifs/biden.gif" height="128" width="128"></a></td>
        </tr>
    </table>
    <table class="images" width="100%"  style="border:0px solid white; width:100%;">
        <tr style="border: 0px;">
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/f1545b9e7ea74f9e802b/?dl=1"><img src="documents/gifs/obama.gif" height="128" width="128"></a></td>
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/ba379b9a5c384722939c/?dl=1"><img src="documents/gifs/wojtek_1.gif" height="128" width="128"></a></td>
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/0f81a65cbdee4e01bfae/?dl=1"><img src="documents/gifs/malte_1.gif" height="128" width="128"></a></td>
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/ae5a0b3ae4c84c25944c/?dl=1"><img src="documents/gifs/bala.gif" height="128" width="128"></a></td>
            <td style="border: 0px;"><a href="https://keeper.mpdl.mpg.de/f/ece2fc07bbee424f80c6/?dl=1"><img src="documents/gifs/person_0004.gif" height="128" width="128"></a></td>
        </tr>
    </table>
</div>

### Dataset Generation

For the input generation, a conda environment is needed, and a few other repositories. Simply run `install.sh` from [scripts](https://github.com/Zielon/INSTA/tree/master/scripts) folder to prepare the workbench.

Next, you can use [Metrical Photometric Tracker](https://github.com/Zielon/metrical-tracker) for the tracking of a sequence. After the processing is done run the `generate.sh` script to prepare the sequence. As input please specify the absolute path of the tracker output.

**For training we recommend at least 1000 frames.**

```shell
# 1) Run the tracker for a selected actor
python tracker.py --cfg ./configs/actors/duda.yml

# 2) Generate a dataset using the script. Importantly, use the absolute path to tracker input and desired output.
./generate.sh /metrical-tracker/output/duda INSTA/data/duda 100

# ./generate.sh {input} {output} {# of test frames from the end}
```

### Citation

If you use this project in your research please cite INSTA:

```bibtex
@proceedings{INSTA:CVPR2023,
  author = {Zielonka, Wojciech and Bolkart, Timo and Thies, Justus},
  title = {Instant Volumetric Head Avatars},
  journal = {Conference on Computer Vision and Pattern Recognition},
  year = {2023}
}
```
