FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# go to workdir
WORKDIR /workdir

# copy project files
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt