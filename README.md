<div align="center">
  <!-- <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h1 align="center" style="font-size: 30px;">Decictor: Towards Evaluating the Robustness of Decision-Making in Autonomous Driving Systems</h1>

[//]: # (  <p align="center">)

[//]: # (    <b>Towards Evaluating the Robustness of Decision-Making in Autonomous Driving Systems</b>)

[//]: # (    <!-- <br /> -->)

[//]: # (    <!-- <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>)

[//]: # (    <br /> -->)

[//]: # (  </p>)
</div>

## About
This repository contains the official implementation of our ICSE 2025 paper, "Decictor: Towards Evaluating the Robustness of Decision-Making in Autonomous Driving Systems." 
We also include existing baselines in our framework to facilitate the comparison and evaluation of different methods. 


## Requirements
### Hardware
Our experiments are conducted on a server with the following specifications:
- CPU: AMD EPYC 7543P 32-Core Processor
- GPU: NVIDIA RTX A5000
- RAM: 256GB
- Storage: 1TB

From experience, we recommend using a machine with minimum specifications of:
- CPU: Intel Core i7-7700K
- GPU: NVIDIA GeForce GTX 1080
- RAM: 32GB
- Storage: 256GB

### Software
Our tool only supports Linux systems. Our experiments are conducted with the following software configurations:
- OS: Ubuntu 20.04
- Anaconda (Installation instructions can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))
- NVIDIA Driver (Installation instructions can be found [here](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/))
- Docker (Installation instructions can be found [here](https://docs.docker.com/engine/install/ubuntu/))
- NVIDIA Container Toolkit (Installation instructions can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

Make sure you have correct software installed before setting up for Decictor. We recommend following the installation instructions from Baidu Apollo to set up the environment. The instructions can be found [here](https://github.com/ApolloAuto/apollo/blob/r7.0.0/docs/specs/prerequisite_software_installation_guide.md).

[//]: # (## Sections)

[//]: # (We offer multiple testing platforms. Please refer to the detailed section for usage instructions.)

[//]: # (1. [ApolloSim]&#40;&#41;)

[//]: # (2. [Carla]&#40;&#41;)

## Setup
We assume the root directory is `/workspace`. Please replace it with your own directory.

### A. Install Baidu Apollo
#### A-1: Download Apollo
1. **Recommended** (Zenodo)    
For convenient configuration, we recommend downloading Apollo from [Zenodo](https://zenodo.org/records/14752133).  
We have fixed several build bugs in Apollo v7.0.0 and added command-line tools for SimControl.  
```shell
cd /workspace
# download apollo.zip
unzip apollo.zip
cd /workspace/apollo
```

2. **Step-by-Step** (If the recommended way does not work)  
Step 1: Download Apollo from the official repository:
```aiignore
git clone -b r7.0.0 https://github.com/ApolloAuto/apollo.git
cd /workspace/apollo
```
Step 2: However, there may be some issues during the building process of Apollo v7.0.0. These can be resolved by applying the following fixes:
```aiignore
vim WORKSPACE
# paste the following patch to line 60
http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "629380c90a77b964d896ed37163f5c3a34f6e6d897311f1df2a7016355c45eff",
    strip_prefix = "zlib-1.2.11",
    urls = ["https://github.com/madler/zlib/archive/v1.2.11.tar.gz"],
)
```
Please refer to https://github.com/ApolloAuto/apollo/issues/14374 and https://github.com/ApolloAuto/apollo/pull/14387/files

Step 3: Move the `Decictor/apollo/sim_control` directory to the `apollo/modules` folder.

##### A-2. Build Apollo

(1) Start docker container
```aiignore
cd /workspace/apollo
bash docker/scripts/dev_start.sh
```
Once finished, you will see a docker container named `apollo_dev_$USER`. (`$USER` is your username)

(2) Compile Apollo
```aiignore
cd /workspace/apollo
bash docker/scripts/dev_into.sh
```
You are now in the container, shown like `$USER@in-dev-docker:/apollo`. You need to run the following command to build Apollo:
```aiignore
./apollo.sh build
```
If the build finishes successfully, you can see the following message:
```aiignore
[ OK ] Done building apollo. Enjoy!
```
You can exit the container by typing `exit`.

### B: Setup Decictor
#### B-1: Download Decictor
We provide two sources (Zenodo and Github) for Decictor. 
Considering that Decictor includes some large external files (e.g., maps), we recommend users start with the **[Zenodo Source](https://zenodo.org/records/14752133)**, which contains all the necessary files.
If you prefer to use the **GitHub Source**, you will need to download the maps separately.

#### 1. Zenodo Source
```aiignore
cd /workspace
# download Decictor.zip
unzip Decictor.zip
```

#### 2. Github Source
```aiignore
cd /workspace
git clone https://github.com/MingfeiCheng/Decictor.git
# download data
cd /workspace/Decictor
# download data.zip from https://drive.google.com/file/d/1j4XVUTicmDR6x5YK_-Rj3ee3PpNRQuSi/view?usp=drive_link
unzip data.zip # note that, the data should be /workspace/Decictor/data
```

##### B-2. Setup Python Environment  
Note: Make sure you have installed Anaconda before running the following commands.  
```aiignore
conda create -n decictor python=3.7.16 -y
conda activate decictor
pip install -r requirements.txt
# install pytorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Usage

### Path Settings
Before using the system, configure the root directories in `config/common/project.yaml` as follows:
```yaml
apollo_root: [Specify the path to the Apollo root directory]
project_root: [Specify the path to the Decictor root directory]
output_root: [Specify the path to the output directory]
```
### Running
After setting up the configuration, execute the following command to start Decictor:
```aiignore
python main_fuzzer.py fuzzer=decictor fuzzer.run_hour=4 seed_name=scenario_1 map_name=sunnyvale_loop run_name=run_1
```
We provide the following seed corpus:

| Seed Name      | Map Name           | 
|----------------|--------------------|
| scenario_1     | sunnyvale_loop     |
| scenario_2     | sunnyvale_loop     |
| scenario_3     | sunnyvale_loop     |
| scenario_4     | sunnyvale_loop     |
| scenario_5     | sunnyvale_big_loop |
| scenario_6     | sunnyvale_big_loop |

We provide the following baselines you can use by setting the `fuzzer` parameter:
- `decictor`
- `drivefuzzer`
- `avfuzzer`
- `behavexplor`
- `samota`
- `deepcollision`
- `random`
- `random_delta`

For convenience, experiment scripts are available in the `scripts` folder:
- To run experiments for RQ1: `bash scripts/rq1.sh`
- To run experiments for RQ2: `bash scripts/rq2.sh`  

Note that you need to update the `project_root` variable in each script before execution.


### Others
Moreover, for the baselines `DoppelTest` and `scenoRITA`, we only modify the initial seeds in their original projects. 
The modified projects are available on [Zenodo](https://zenodo.org/records/14752133). Please refer to their documentation and follow the provided instructions to set up the projects.
We provide a `run_scenarios.sh` script to run their projects under Decictor's settings. To use it, simply update the `seed_root` variable in the shell script.


## Citation & Contact
```aiignore
@article{cheng2024evaluating,
  title={Decictor: Towards Evaluating the Robustness of Decision-Making in Autonomous Driving Systems},
  author={Cheng, Mingfei and Zhou, Yuan and Xie, Xiaofei and Wang, Junjie and Meng, Guozhu and Yang, Kairui},
  journal={arXiv preprint arXiv:2402.18393},
  year={2024}
}
```
Contact: [Mingfei Cheng](snowbirds.mf@gamil.com)

## Contribution
We warmly welcome contributions and suggestions to enhance this project and promote research in industrial-grade ADS testing. Your contributions are highly valued and greatly appreciated.

## License
Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Acknowledgements
We thanks for the following open-source projects:  
* [DoppelTest](https://github.com/Software-Aurora-Lab/DoppelTest)   
* [scenoRITA-7.0
](https://github.com/Software-Aurora-Lab/scenoRITA-7.0)  
* [ScenarioRunner](https://github.com/carla-simulator/scenario_runner)