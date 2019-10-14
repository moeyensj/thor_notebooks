# THOR Notebooks
Jupyter Notebooks for Tracklet-less Heliocentric Orbit Recovery  
[![Docker Pulls](https://img.shields.io/docker/pulls/moeyensj/thor_notebooks)](https://hub.docker.com/r/moeyensj/thor_notebooks)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Installation

The THOR code repository can be found at: https://github.com/moeyensj/thor

Please follow the installation instructions included with the THOR code repository. Then proceed along one of the following installation paths corresponding to the chosen THOR code installation path.

### Source
Clone this repository using either `ssh` or `https` into the same directory as the THOR code. For example:

```
ls projects/thor/
    thor
    thor_notebooks
```

Activate the conda enviroment in which THOR dependencies were installed, then install additional dependencies:  
```conda install -c defaults -c conda-forge --file additional_requirements.txt```

To install pre-requisite software using pip:  
```pip install -r additional_requirements.txt```

### Docker

A Docker container with the latest version of the THOR notebooks and code can be pulled using:  
```docker pull moeyensj/thor_notebooks:latest```

To run the container (feel free to map the ports differently):  
```docker run -it --rm -p 1719:1719 moeyensj/thor_notebooks:latest```