"""This holds dictionary to various datasets. when adding a new dataset, add it to this dictionary!"""


from datasets.dataset_generic import DatasetGeneric
from datasets.dataset_aspire import DatasetAspire
from datasets.dataset_io import DatasetIO


DATASET_INDEX = {
    'generic': DatasetGeneric,
    'aspire': DatasetAspire,
    'torchio': DatasetIO,
}
