Code to accompany Harvey Merton and Thomas Delamore's final year research project at UoA (see: XXX #paper location#). This code is built off various repositories but mostly the FSSIM (https://github.com/AMZ-Driverless/fssim) and fsd_skeleton (https://github.com/AMZ-Driverless/fsd_skeleton) repositories by AMZ Driverless. This README file is based off the README in the later repository as this repository forms the higher-level structure of the workspace.


# Repository organisation

The code is organised in several top level packages/directories. The top level should adhere to the following subdivision of functionality (a more detailed description can be found in the folders themselves):

**fsd_common/** - top-level launchfiles, and other files used by many packages

**perception/** - folder for perception packages

**estimation/** - folder for estimation packages

**control/** - folder for control packages

- - - -

# Placement of ROS packages
ROS Packages should be added in one of the top level work-package folders. The top level work-package folders themselves should not be used to store ros package information. The directory tree for lidar would look like:

```
~/fsd_skeleton
  |__ environment.sh
  |__ update_dependencies.sh
  |__ src
      |__ 0_fsd_common
      |   |__ fsd_common_meta
      |   |   |__ missions
      |   |__ fsd_common_msgs
      |__ 1_perception
      |   |__ perception_meta
      |   |__ lidar_cone_detection
      |   |   |__ package.xml
      |   |   |__ CMakeLists.txt
      |   |   |__ ...
      |
      |__ 2_estimation
      |   |__ estimation_meta
      |   |__ velocity_estimator
      |
      |__ 3_control
```
- - - -

# Aliases (useful commands)
Once running `update_dependencies`, some aliases for commands will be added. Restart the terminal and the following commands will be available:
* `FSD_source`: sources environment from anywhere
* `FSD_cd`: change directory to root directory of skeleton_repo
* `FSD_build`: clean and build project (catkin clean and catkin build)
* `FSD_launch_acceleration`: launch mission, e.g. acceleration, trackdrive, autox etc
* `FSD_rviz_acceleration`: launch RVIZ with custom config for mission, e.g. acceleration, trackdrive, autox etc 
* `FSD_ATS`: run automated test

Look at`fsd_aliases` to see full list, or add more custom aliases.
- - - -

# Setting up the Workspace
**1 Clone the repository:**
```
cd ~
git clone 
```
**2 Install dependencies**
```
cd ~/uoa_fsd
./update_dependencies.sh
```
Add extra dependencies: 
pip install -r requirements.txt


**3 Build workspace**
```
cd ~/uoa_fsd
catkin build
```

**4 Source environment**

Assuming you've run `./update_dependencies.sh` succesfully and restarted the terminal.
```
FSD_source
```
Else,
```
cd ~/uoa_fsd
source devel/setup.bash
```

TIP: to avoid having to source the workspace upon every terminal launch, add 'source devel/setup.bash' to the .bashrc file (hidden file located in ~/ directory)

**5 Test setup**
```
roslaunch fssim_interface fssim.launch
```
in new terminal
```
roslaunch fsd_common_meta trackdrive.launch
```
You should see the FSSIM open and a green line appear on the track to mark the trajectory. If the sim is setup for pure pursuit, the vehicle will follow the green line.
- - - -

# Conventions
- - - -
## ROS naming conventions
We use the naming conventions defined at http://wiki.ros.org/ROS/Patterns/Conventions
### Work packages:
`work_package`, lowercase and `_` as separator, e.g. `lidar`.
### ROS packages:
`workpackage_somename`, lowercase and `_` as separator, e.g. `lidar_trimmer`, as to make it clear what the package is used for.
### ROS nodes
`node_name`, lowercase and `_` as separator. Can be short.
### ROS topics
`topic_name`, lowercase and `_` as separator.
### ROS messages
`CamelCased.msg` for message filenames. Message types are always CamelCase, whereas message fields are lowercase and `_` as separator, e.g.
```
MyMessage.msg:
Header header
Float64 my_float
geometry_msgs/Point my_point
```

## Style guides
### ROS C++:
Google Style (http://wiki.ros.org/CppStyleGuide)

* Files: `under_scored`, exception for `.msg` files, `CMakeLists.txt`.
* Classes/types: `CamelCase`
* Functions/methods: `camelCase`
* Variables: `under_scored` and DESCRIPTIVE.
* Constants: `ALL_CAPITALS`.
* Global variables: AVOID except special cases. Rather have parameters defined in `config.yaml`.

### ROS Python
PEP-8 style (http://wiki.ros.org/PyStyleGuide)

### README files
Markdown syntax (https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

# Future improvements

* Update to Python3, ROS Noetic and Ubuntu Bionic
