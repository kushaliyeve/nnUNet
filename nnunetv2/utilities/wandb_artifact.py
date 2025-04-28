import logging
import shutil
import string
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

# import boto3
import wandb

from nnunetv2.utilities.misc import generate_id

# from .s3 import download_files, get_bucket_and_key, upload_files

# copied from ai_ovai repo

logger = logging.getLogger(__name__)


def file_uri_to_path(uri: str) -> str:
    """Convert url with file scheme.

    As allowed by the RFC, URLs with file scheme allows:
        * file:/path (no hostname)
        * file://hostname/path
        * file:///path (empty hostname)

    This function helps to handle these cases and always return a relative
    or absolute local filepath.

    Args:
        uri (str): uri with file: scheme to convert to a local filepath

    Returns:
        str: string
    """

    parsed = urlparse(uri)
    if parsed.scheme != "file":
        msg = f"uri must use file: scheme. Got {uri}"
        raise ValueError(msg)
    path = parsed.path if parsed.netloc is None else parsed.netloc + parsed.path

    return path


def retrieve_artifact(artifact_uri, output_dirpath=".", type: str = None, api=None):
    artifact_uri = str(
        artifact_uri,
    )  # TODO handle ArtifactUri pydantic model instead of casting to str
    # if Path(artifact_uri).suffix:
    #     msg = "The provided uri is a path to a specific file. You can provide only the path to a data directory object."
    #     raise ValueError(msg)

    parsed_uri = urlparse(artifact_uri)

    # if parsed_uri.scheme == "s3":
    #     bucket, key = get_bucket_and_key(artifact_uri)
    #     data_directory_path = download_files(
    #         bucket,
    #         key,
    #         basepath=output_dirpath,
    #         keep_prefix=True,
    #         keep_only_last_segment=True,
    #     )
    if parsed_uri.scheme == "wandb":
        artifact_uri = str(Path(parsed_uri.netloc) / parsed_uri.path.lstrip("/"))
        try:
            if not wandb.run:
                wandb_api = api if api else wandb.Api()
                artifact = wandb_api.artifact(artifact_uri, type=type)
            else:
                artifact = wandb.run.use_artifact(artifact_uri, type=type)

            logger.info(f"Downloading {type} artifact from : {artifact_uri}")
            data_directory_path = artifact.download(
                Path(output_dirpath)
                / artifact_uri.split("/")[-1],  # just keep artifact name ignoring entity/project/
            )

        except Exception as e:
            msg = f"Error while trying to download an artifact from WB. Error {e}"
            raise ValueError(msg)

    elif parsed_uri.scheme == "file" or parsed_uri.scheme == "":
        data_directory_path = (
            Path(file_uri_to_path(artifact_uri))
            if parsed_uri.scheme == "file"
            else Path(parsed_uri.path)
        )
        # TODO make this behaviour the same as log_artifact
        logger.debug(f"{artifact_uri} is a local filepath. Ignoring {output_dirpath}")

        if not data_directory_path.exists():
            msg = f"Provided uri {artifact_uri} seem a local path but cannot be locally retrieved."
            raise FileNotFoundError(msg)
    else:
        msg = f"Invalid uri {artifact_uri}, scheme : {parsed_uri.scheme}. Supported schemes: s3, file or no scheme"
        raise ValueError(msg)

    logger.info(f"Retrieved {type} from {artifact_uri}, local path : {data_directory_path}")

    return data_directory_path


def log_artifact(
    src_dirpath: str,
    dest_uri: str,
    type: str,
    name: Union[str, string.Template] = string.Template("run-$run_id-$type"),
    log_to_wandb: bool = True,
    aliases: List[str] = None,
    # s3: Optional[boto3.resources.base.ServiceResource] = None,
) -> Union[str, wandb.Artifact]:
    """Transfer local data to destination uri and log it as a wandb's Artifact.

    Note:
        Even if specifying a local path as dest_uri or skipping wandb artifact logging seems pointless,
        these behaviours could be useful for debugging/development use cases. However, dest_uri should always point
        to some remote storage as well as keeping `log_to_wandb=True`, when using this function in production-like
        environments.

    Args:
        src_dirpath (str): path to the local directory where artifact data is stored.
        dest_uri (str): uri where data will be transfered. Supported schemes: file and s3.
                        Using file:// scheme, the directory tree below src_dirpath will be
                        copied to dest_uri.
        type (str): artifact type (eg. dataset, model, etc.).
        name (string.Template, optional): artifact name.
                                        Can be formatted using $run_id and $type placeholders, but name must be a string.Template instance.
                                        if the caller doesn't run in a wandb run context, than the run_id will be generated using
                                        `ai_ovai.utils.mics.generate_id` function. Defaults to string.Template("run-$run_id-$type").
        log_to_wandb (bool, optional): True to log it as wandb Artifact for the current active run, False to skip and just move the data
                                    in the dest_uri. The artifacts should always tracked with wandb, however this step can be skipped
                                    for testing purposes. Defaults to True.
        aliases (List[str], optional): list of aliases when logging artifact to wandb. Ignored if log_to_wandb = False. Defaults to None.
        s3 (boto3.resources.base.ServiceResource, optional): boto3 s3 ServiceResource instance used to transfer data to s3. Defaults to None.

    Raises:
        ValueError: if dest_uri scheme is not file:// or s3://

    Returns:
        str: An wandb.Artifact instance  if `log_to_wandb=True`. An Uri str where artifact is stored otherwise.
    """
    run_id = getattr(wandb.run, "id", generate_id())

    if isinstance(name, string.Template):
        name = name.substitute(run_id=run_id, type=type)
    artifact_uri = f"{str(dest_uri).rstrip('/')}/{name}"

    url = urlparse(artifact_uri)
    # if url.scheme == "s3":
    #     artifact_bucket, artifact_key = get_bucket_and_key(artifact_uri)
    #     upload_files(src_dirpath, artifact_bucket, artifact_key, keep_directory=False, s3=s3)
    if url.scheme == "file":
        dest_path = file_uri_to_path(artifact_uri)
        shutil.copytree(src_dirpath, dest_path, dirs_exist_ok=True)
    else:
        msg = "dest_uri must be an URI. To reference a local filepath use file://"
        raise ValueError(msg)

    if log_to_wandb:
        artifact = wandb.Artifact(name, type=type)
        artifact.add_reference(artifact_uri)
        wandb.log_artifact(artifact, aliases=aliases)

    return artifact if log_to_wandb else artifact_uri
