"""This holds dictionary to various datasets. when adding a new dataset, add it to this dictionary!"""



from datasets.dataset_generic import DatasetGeneric
from datasets.dataset_aspire import DatasetAspire


DATASET_INDEX = {
    'generic': DatasetGeneric,
    'aspire': DatasetAspire

}