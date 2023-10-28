from dicee.executer import Execute
import pytest
from dicee.config import Namespace
class TestBackends:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = Namespace()
        args.dataset_dir = 'KGs/UMLS'
        args.backend = 'pandas'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_rdf_as_backend(self):
        args = Namespace()
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend = 'rdflib'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = Namespace()
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'pandas'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = Namespace()
        args.dataset_dir = 'KGs/UMLS'
        args.backend = 'polars'
        Execute(args).start()
