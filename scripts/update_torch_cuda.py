import logging
import subprocess
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


console = Console()

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
log = logging.getLogger("rich")


def is_cuda_available() -> bool:
    """Check if CUDA is available, without global torch import."""
    import torch

    return torch.cuda.is_available()


def get_torch_version() -> tuple[str, Optional[str]]:
    """Get the current version of PyTorch, without global torch import.

    If it is for a specific version, it will have the '+cu11x' suffix.
    """
    import torch

    version = torch.__version__

    log.info(f"Current `torch` verison: {version}")

    if "+" not in version:
        return version, None

    return version.split("+")[0], version.split("+")[1]


def get_torchvision_version() -> tuple[str, Optional[str]]:
    """Get the current version of torchvision, without global importing.

    If it is for a specific version, it will have the '+cu11x' suffix.
    """
    import torchvision

    version = torchvision.__version__

    log.info(f"Current `torchvision` verison: {version}")

    if "+" not in version:
        return version, None

    return version.split("+")[0], version.split("+")[1]


def get_supported_cuda_version() -> float:
    """Get the highest supported CUDA version.

    NVIDIA-SMI seems to be the most consistent version display, so needed to use subprocess to get
    it out.
    """
    nvidia_smi_process = subprocess.Popen(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    nvidia_smi_stdout, _ = nvidia_smi_process.communicate()

    cuda_version_line = nvidia_smi_stdout.decode().split("\n")[2]
    cuda_version_list = list(filter(None, cuda_version_line.split(" ")))
    cuda_version = float(cuda_version_list[-2])

    log.info(f"Supported CUDA version: {cuda_version}")

    return cuda_version


def get_torch_cuda_suffix(cuda_version: float) -> str:
    """Get the relevant CUDA suffix from the version."""
    if cuda_version > 11 and cuda_version < 11.3:
        return "cu111"
    if cuda_version > 11.3:
        return "cu113"

    return ""


def update_torch(torch_version: tuple[str, str], torchvision_version: tuple[str, str]) -> None:
    """Install the desired torch and torchvision versions."""
    torch_package_version = f"{torch_version[0]}+{torch_version[1]}"
    torchvision_package_version = f"{torchvision_version[0]}+{torchvision_version[1]}"
    log.info(
        f"Installing torch=={torch_package_version} and torchvision=={torchvision_package_version}"
    )

    process_args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch=={torch_package_version}",
        f"torchvision=={torchvision_package_version}",
        "-f",
        "https://download.pytorch.org/whl/torch_stable.html",
    ]

    with subprocess.Popen(
        process_args, stdout=subprocess.PIPE, universal_newlines=True
    ) as process:
        if process.stdout is not None:
            for line in iter(process.stdout.readlines()):
                console.log(line)


def main(force_update: bool = False) -> None:
    """Run the script."""
    if not is_cuda_available():
        log.warning("CUDA is not available. Are you sure about running this script?")
        return

    if force_update:
        log.info("`force_update` is True")

    should_update = force_update

    log.debug("Get supported CUDA version")
    supported_cuda_version = get_supported_cuda_version()

    log.debug("Getting ideal torch suffix.")
    torch_suffix = get_torch_cuda_suffix(supported_cuda_version)

    log.debug("Getting torch version")
    torch_version = get_torch_version()

    log.debug("Getting torchvision version")
    torchvision_version = get_torchvision_version()

    if torch_version[1] != torch_suffix or should_update:
        should_update = True
        torch_version = (torch_version[0], torch_suffix)

    if torchvision_version[1] != torch_suffix or should_update:
        should_update = True
        torchvision_version = (torchvision_version[0], torch_suffix)

    if should_update:
        log.debug("Updating torch and torchvision")
        update_torch(torch_version, torchvision_version)  # type: ignore[arg-type]

    if not should_update:
        log.info(
            "No update to torch is needed. The best wheel for the current CUDA version is already installed."
        )

    log.info("Done")


if __name__ == "__main__":
    main(True)
