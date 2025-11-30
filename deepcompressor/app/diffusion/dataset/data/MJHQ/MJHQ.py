import json
import os
import random

import datasets
import zipfile
import tempfile
from PIL import Image

_CITATION = """\
@misc{li2024playground,
      title={Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation},
      author={Daiqing Li and Aleks Kamko and Ehsan Akhgari and Ali Sabet and Linmiao Xu and Suhail Doshi},
      year={2024},
      eprint={2402.17245},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_DESCRIPTION = """\
We introduce a new benchmark, MJHQ-30K, for automatic evaluation of a model’s aesthetic quality.
 The benchmark computes FID on a high-quality dataset to gauge aesthetic quality.
"""

_HOMEPAGE = "https://huggingface.co/datasets/playgroundai/MJHQ-30K"

_LICENSE = (
    "Playground v2.5 Community License "
    "(https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md)"
)

IMAGE_URL = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/mjhq30k_imgs.zip"

META_URL = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/meta_data.json"


class MJHQConfig(datasets.BuilderConfig):
    def __init__(self, max_dataset_size: int = -1, return_gt: bool = False, **kwargs):
        super(MJHQConfig, self).__init__(
            name=kwargs.get("name", "default"),
            version=kwargs.get("version", "0.0.0"),
            data_dir=kwargs.get("data_dir", None),
            data_files=kwargs.get("data_files", None),
            description=kwargs.get("description", None),
        )
        self.max_dataset_size = max_dataset_size
        self.return_gt = return_gt


class DCI(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = MJHQConfig
    BUILDER_CONFIGS = [MJHQConfig(name="MJHQ", version=VERSION, description="MJHQ-30K full dataset")]
    DEFAULT_CONFIG_NAME = "MJHQ"

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "category": datasets.Value("string"),
                "image": datasets.Image(),
                "prompt": datasets.Value("string"),
                "prompt_path": datasets.Value("string"),
                "image_root": datasets.Value("string"),
                "image_path": datasets.Value("string"),
                "split": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        # Prefer local Hugging Face cache if present to avoid re-downloading.
        local_cache = os.path.expanduser("~/.cache/huggingface/hub/datasets--playgroundai--MJHQ-30K")
        meta_path = None
        image_root = None

        if os.path.exists(local_cache):
            # Try to locate meta_data.json anywhere under the cache directory
            for root, dirs, files in os.walk(local_cache):
                if "meta_data.json" in files:
                    meta_path = os.path.join(root, "meta_data.json")
                    break

            # Try to find any .jpg file and assume its parent is a category folder
            # so the image_root is the parent directory of that category.
            jpg_path = None
            for root, dirs, files in os.walk(local_cache):
                for f in files:
                    if f.lower().endswith(".jpg"):
                        jpg_path = os.path.join(root, f)
                        break
                if jpg_path:
                    break

            if jpg_path:
                # root currently points to the category folder (e.g., .../mjhq30k_imgs/<category>)
                image_root = os.path.dirname(jpg_path)
                # move up one level to reach the directory that contains category folders
                image_root = os.path.dirname(image_root)

            # If images are present as a zip file, extract to a temp dir and use that
            if image_root is None:
                for root, dirs, files in os.walk(local_cache):
                    for f in files:
                        if f.lower().endswith("mjhq30k_imgs.zip") or f.lower().endswith(".zip"):
                            zip_file = os.path.join(root, f)
                            try:
                                tmpdir = tempfile.mkdtemp(prefix="mjhq30k_")
                                with zipfile.ZipFile(zip_file, "r") as zf:
                                    zf.extractall(tmpdir)
                                # try to find extracted folder containing images
                                for eroot, edirs, efiles in os.walk(tmpdir):
                                    for ef in efiles:
                                        if ef.lower().endswith(".jpg"):
                                            image_root = os.path.dirname(eroot)
                                            break
                                    if image_root:
                                        break
                            except Exception:
                                image_root = None
                            if image_root:
                                break
                    if image_root:
                        break

        # Fall back to downloading if we couldn't find local files
        if meta_path is None:
            meta_path = dl_manager.download(META_URL)
        if image_root is None:
            image_root = dl_manager.download_and_extract(IMAGE_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"meta_path": meta_path, "image_root": image_root}
            ),
        ]

    def _generate_examples(self, meta_path: str, image_root: str):
        with open(meta_path, "r") as f:
            meta = json.load(f)

        names = list(meta.keys())
        if self.config.max_dataset_size > 0:
            random.Random(0).shuffle(names)
            names = names[: self.config.max_dataset_size]
            names = sorted(names)

        for i, name in enumerate(names):
            category = meta[name]["category"]
            prompt = meta[name]["prompt"]
            image_path = os.path.join(image_root, category, f"{name}.jpg")
            yield (
                i,
                {
                    "filename": name,
                    "category": category,
                    "image": Image.open(image_path) if self.config.return_gt else None,
                    "prompt": prompt,
                    "meta_path": meta_path,
                    "image_root": image_root,
                    "image_path": image_path,
                    "split": self.config.name,
                },
            )
