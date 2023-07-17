from dicee.executer import Execute
import pytest
from dicee.config import Namespace
class TestPolyak:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = Namespace()
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'pandas'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_rdf_as_backend(self):
        args = Namespace()
        args.path_dataset_folder = 'KGs/Family'
        args.backend = 'pandas'
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
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'polars'
        Execute(args).start()
