from __future__ import absolute_import

import importlib
import os
import yaml
import gorilla

import pandas as pd
import pickle

import paddle
import paddle.fluid as fluid

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import try_mlflow_log
import numpy as np

FLAVOR_NAME = "paddle"
_MODEL_SAVE_PATH = "model"


def get_default_conda_env():
    return _mlflow_conda_env(
        additional_conda_deps=[
            "paddle=={}".format(paddle.__version__),
        ],
        additional_pip_deps=None, #["cloudpickle=={}".format(cloudpickle.__version__)],
        additional_conda_channels=["paddle",])


def save_model(path, feeded_var_names, target_vars, executor, conda_env=None, mlflow_model=Model(),
               **kwargs): 
    # paddle_module = importlib.import_module(paddle_module)

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    #model_subpath = "data"
    #model_path = os.path.join(path, model_subpath)
    os.makedirs(path)
    #os.makedirs(model_path)
    #print("paddle.py save_model path: ", path)
    fluid.io.save_inference_model(dirname=path, feeded_var_names=feeded_var_names,
    target_vars=target_vars, executor=executor)
    
    mlflow_model.add_flavor(FLAVOR_NAME,
                            paddle_version=paddle.__version__,)
                            #data=path)
    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.paddle",
                        env=conda_env_subpath)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(artifact_path, conda_env=None,
              **kwargs):
    Model.log(artifact_path=artifact_path, flavor=mlflow.paddle,
              conda_env=conda_env, **kwargs)


def load_model(model_uri, executor, **kwargs): # require executor
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    paddle_model_artifacts_path = os.path.join(
        local_model_path,
        flavor_conf.get("data", _MODEL_SAVE_PATH))
    return _load_model(paddle_model_artifacts_path, executor) 


def _load_model(path, executor, **kwargs): # require executor
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=executor)
    return inference_program, feed_target_names, fetch_targets, executor


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``pytorch`` flavor.
    """
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    inference_program, feed_target_names, fetch_targets, executor = _load_model(path, executor = exe, **kwargs)
    return _PaddleWrapper(inference_program=inference_program, feed_target_names=feed_target_names, fetch_targets=fetch_targets, executor=executor)


class _PaddleWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """
    def __init__(self, inference_program, feed_target_names, fetch_targets, executor):
        self.inference_program = inference_program
        self.feed_target_names = feed_target_names
        self.fetch_targets = fetch_targets
        self.executor = executor

    def predict(self, data):
        exe = self.executor
        generated = {column:np.asarray([data[column].values.tolist()]).astype("float32") for column in data.columns}
        results = exe.run(self.inference_program,
              feed=generated,
              fetch_list=self.fetch_targets)
        print(type(results))
        print(results)
        predicted = pd.DataFrame(results[0])
        print(predicted)
        return predicted
