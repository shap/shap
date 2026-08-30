#!/bin/bash

set -e
set -x

if [ "$GITHUB_EVENT_NAME" == "schedule" ]; then
    # todo: is this correct or do we need scientific-python-nightly-wheels
    ANACONDA_ORG="scipy-wheels-nightly"
    ANACONDA_TOKEN="$SHAP_NIGHTLY_UPLOAD_TOKEN"

    # Install Python 3.8 because of a bug with Python 3.9
    export PATH=$CONDA/bin:$PATH
    conda create -n upload -y python=3.11
    source activate upload
    conda install -y anaconda-client

    # Force a replacement if the remote file already exists
    anaconda -t $ANACONDA_TOKEN upload --force -u $ANACONDA_ORG $ARTIFACTS_PATH/*
    echo "Index: https://pypi.anaconda.org/$ANACONDA_ORG/simple"
end
