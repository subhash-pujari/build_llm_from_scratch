# build_llm_from_scratch


Setting up python virtual env.

```
python -m venv .venv
.venv\Scripts\activate
```

### Setting up docker

```
docker pull continuumio/miniconda3

docker run  -i -t -v "//c/Subhash/code/:/home/subhash/code/" continuumio/miniconda3 /bin/bash

<!--  create conda env and install required packages inside the docker image  -->

docker commit <container-name> ubuntu-build-llm

docker run  -i -t -v "//c/Subhash/code/:/home/subhash/code/" ubuntu-build-llm
```
