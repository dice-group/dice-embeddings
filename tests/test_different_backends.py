from dicee.executer import Execute, get_default_arguments
import pytest


class TestPolyak:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = get_default_arguments([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'pandas'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_rdf_as_backend(self):
        args = get_default_arguments([])
        args.path_dataset_folder = 'KGs/Family'
        args.backend = 'pandas'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = get_default_arguments([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'modin'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_modin_rdf_as_backend(self):
        args = get_default_arguments([])
        args.path_dataset_folder = 'KGs/Family'
        args.backend = 'modin'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = get_default_arguments([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'polars'
        Execute(args).start()
