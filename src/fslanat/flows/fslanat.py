import shutil
import subprocess
import tempfile
from pathlib import Path

import prefect

from fslanat.models.fslanat import FSLAnatResult


def _img_stem(img: Path) -> str:
    return img.name.removesuffix(".gz").removesuffix(".nii")


def _predict_fsl_anat_output(out: Path, basename: str) -> Path:
    return (out / basename).with_suffix(".anat").absolute()


@prefect.task
def _fslanat(image: Path, out: Path):
    basename = _img_stem(image)
    anat = _predict_fsl_anat_output(out, basename)

    # if the output already exists, we don't want this to run again.
    # fsl_anat automatically and always adds .anat to the value of -o, so we check for
    # the existence of that predicted output, but then feed in the unmodified value of
    # -o to the task
    if not anat.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfsl = Path(tmpdir) / basename
            subprocess.run(
                ["fsl_anat", "-i", f"{image}", "-o", f"{tmpfsl}"],
                capture_output=True,
            )
            tmpout = _predict_fsl_anat_output(Path(tmpdir), basename)
            FSLAnatResult.from_root(tmpout)
            shutil.copytree(tmpout, anat)


@prefect.flow
def fslanat_flow(images: frozenset[Path], out: Path) -> None:
    _fslanat.map(images, out=out)  # type: ignore
