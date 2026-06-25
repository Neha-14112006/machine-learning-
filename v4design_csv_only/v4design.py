import csv
import datasets
import os

_DESCRIPTION = "example"

_HOMEPAGE = "TODO"

_LICENSE = "TODO"

_URL = "https://zenodo.org/record/4896487/files/V4Design_Europeana_style_dataset.csv?download=1"

class V4Design(datasets.GeneratorBasedBuilder):
    """TODO"""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features(
            {"id": datasets.Value("string"),
            "url": datasets.Value("string"),
            "uri": datasets.Value("string"),
            "style":datasets.ClassLabel(names=['Rococo', 'Baroque', 'Other']),
            "rights": datasets.Value("string"),}
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation="TODO",
        )

    def _split_generators(self, dl_manager):
        csv_file = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"csv_file": csv_file}
            ),
        ]

    def _generate_examples(self, csv_file):
        with open(csv_file, encoding="utf-8") as f:
            data =  csv.DictReader(f)
            for row, item in enumerate(data):
                yield row, {"id": item['id'],"url": item['url'], "uri": item['uri'], "style":item['style'], "rights": item['rights']}
