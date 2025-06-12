import os
import pytest
import shutil
import numpy as np
import pandas as pd
from dicee.config import Namespace
from dicee.executer import Execute
from dicee.knowledge_graph_embeddings import KGE

class TestPredictLitRegression:
    """Regression tests for literal prediction using interactive KGE model Family dataset."""
    
    @pytest.fixture(scope="function", autouse=True)
    def generate_literal_files(self):
        """
        Generate training and testing files for literal triples.

        Args:
            file_path (str): The directory where the files will be saved.
        """

        literal_triples =     [
            ('F5F62', 'Age', 18), ('F1F3', 'Height', 164),
            ('F6M99', 'Height', 138), ('F6F97', 'Age', 17),
            ('F6M92', 'Weight', 80), ('F3F41', 'Age', 59),
            ('F9F141', 'Age', 53), ('F2F33', 'Age', 39),
            ('F2M35', 'Height', 138), ('F3F46', 'Height', 140),
            ('F9M155', 'Weight', 77), ('F6F70', 'Height', 152),
            ('F7F121', 'Age', 91), ('F2M23', 'Weight', 75),
            ('F2F24', 'Weight', 57), ('F9M162', 'Age', 42),
            ('F1F7', 'Weight', 62), ('F5M60', 'Height', 151),
            ('F10F181', 'Weight', 60), ('F3M47', 'Age', 40),
            ('F2M16', 'Height', 163), ('F6F84', 'Height', 149),
            ('F10F198', 'Age', 36), ('F7M125', 'Weight', 40),
            ('F10M173', 'Height', 177)
        ]
        #train test splits
        train_triples = literal_triples[:20]
        test_triples = literal_triples[20:]

        # Output directory
        output_dir =  "KGs/Family/literals"
        os.makedirs(output_dir, exist_ok=True)

        train_df = pd.DataFrame(train_triples)
        test_df = pd.DataFrame(test_triples)
        train_df.to_csv( os.path.join(output_dir, "train.txt"), header=False, sep ="\t",index=False)
        test_df.to_csv( os.path.join(output_dir, "test.txt"), header=False,sep ="\t", index=False)


    @pytest.fixture(scope="class")
    def family_model(self):
        """Setup Keci model trained on Family dataset."""

        # Create training triples for the Family dataset.
        self.create_train_triples(path="KGs/Family")
        
        # Set up the arguments for the Keci model
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.path_single_kg = "KGs/Family/train.txt"
        args.backend = "pandas"
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 64
        args.trainer = 'torchCPUTrainer'  # Force CPU
        
        result = Execute(args).start()
        model = KGE(path=result['path_experiment_folder'])
        
        # Remove the train file if it exists to ensure clean directory
        if os.path.exists(args.path_single_kg):
            os.remove(args.path_single_kg)
        return {
            'model': model
            }
    
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_train_literals(self, family_model):
        """Test training of literal embedding model using interactive KGE model."""
        
        model = family_model['model']
        train_file_path="KGs/Family/literals/train.txt"

        # Train with literals
        model.train_literals(train_file_path=train_file_path,
        eval_litreal_preds=False,
        loader_backend="pandas",
        num_epochs=20,
        batch_size=50, device='cpu')

        # remove literal test artifacts
        shutil.rmtree(os.path.dirname(train_file_path))


    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_predict_literals_single(self, family_model ):
        """Test Literal values prediction ( single subject-predicate pair) using interactive KGE model."""
        model = family_model["model"]
        train_file_path="KGs/Family/literals/train.txt"

        # Train with literals
        model.train_literals(train_file_path=train_file_path,
        eval_litreal_preds=False,
        loader_backend="pandas",
        num_epochs=100,
        batch_size=50, device='cpu')

        # Predict literals for a known entity
        entity = "F6F72"
        attribute = "Age"

        result = model.predict_literals(entity=entity, attribute=attribute)

        assert result.shape == (1,), "Expected array with shape (1,)"
        prediction = result[0]
        assert isinstance(prediction, (int, float)), "Result is not a numeric value"
        assert 44.3 <= prediction <= 44.9, f"Result {prediction} is not within the expected range"

        # remove literal test artifacts
        shutil.rmtree(os.path.dirname(train_file_path))


    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_predict_literals_batch(self, family_model):
        """Test Literal values prediction(batch prediction) using interactive KGE model."""
        
        model = family_model['model']
        train_file_path="KGs/Family/literals/train.txt"

        # Train with literals
        model.train_literals(train_file_path=train_file_path,
        eval_litreal_preds=False,
        loader_backend="pandas",
        num_epochs=20,
        batch_size=512)

        # Predict literals for a known entity
        entities   = ['F2M34', 'F9M167', 'F2F24']
        attributes = ['Age', 'Weight', 'Height']

        results = model.predict_literals(entity=entities, attribute=attributes)
        

        # Assert the result is a numpy array of the same size as the input
        assert isinstance(results, np.ndarray), "Results should be a numpy array"
        assert len(results) == len(entities), "Results size does not match input size"

        # Assert all entries are numerical types
        for result in results:
            assert isinstance(result, (int, float)), f"Result {result} is not a numeric value"
        
        # remove literal test artifacts
        shutil.rmtree(os.path.dirname(train_file_path))
        
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_evaluate_literal_prediction(self, family_model):
        """Test Evaluation of Literal Prediction using interactive KGE model."""
        
        model = family_model['model']
        train_file_path="KGs/Family/literals/train.txt"

        # Train with literals
        model.train_literals(train_file_path=train_file_path,
        eval_litreal_preds=False,
        loader_backend="pandas",
        num_epochs=20,
        batch_size=512)

        eval_file_path = "KGs/Family/literals/test.txt"
        
        # Evaluate literal predictions
        lit_prediction_errors = model.evaluate_literal_prediction(
            eval_file_path=eval_file_path,
            store_lit_preds = False,
            eval_literals=True,
            loader_backend = "pandas",
            return_attr_error_metrics = True)

        # Assert the result is a DataFrame
        assert isinstance(lit_prediction_errors, pd.DataFrame), "Results should be a DataFrame"
        assert not lit_prediction_errors.empty, "Results DataFrame should not be empty"
        assert not lit_prediction_errors.isnull().values.any() , "Results DataFrame should not contain NaN values"

        # remove literal test artifacts
        shutil.rmtree(os.path.dirname(eval_file_path))

    def create_train_triples(self, path : str = None):
        """
        Create training triples for the Family dataset.

        Args:
            path (str): The directory where the files will be saved.
        """
        # List of relation triples
        relation_triples = [
            ('F2M25', 'type', 'NamedIndividual'), ('F2F30', 'type', 'Grandmother'),
            ('F10F201', 'hasParent', 'F10F198'), ('F6M90', 'type', 'Son'),
            ('F2M35', 'hasParent', 'F2M34'), ('F10F179', 'hasParent', 'F10M171'),
            ('F6F96', 'type', 'NamedIndividual'), ('F6M73', 'hasChild', 'F6M75'),
            ('F7F121', 'type', 'Grandmother'), ('F10M173', 'type', 'Father'),
            ('F9M167', 'married', 'F9F168'), ('F2F33', 'type', 'Daughter'),
            ('F5F62', 'type', 'Daughter'), ('F6F79', 'type', 'Person'),
            ('F10F181', 'type', 'Person'), ('F1F3', 'type', 'Female'),
            ('F3F46', 'hasChild', 'F3M47'), ('F7F129', 'married', 'F7M130'),
            ('F6M71', 'hasSibling', 'F6F84'), ('F9M162', 'type', 'NamedIndividual'),
            ('F9M155', 'hasParent', 'F9F154'), ('F6F87', 'type', 'Sister'),
            ('F3F41', 'married', 'F3M40'), ('F1F7', 'type', 'Female'),
            ('F5M60', 'type', 'Grandfather'), ('F2M23', 'married', 'F2F24'),
            ('F6M99', 'type', 'Person'), ('F6F97', 'type', 'Daughter'),
            ('F6M92', 'hasParent', 'F6F70'), ('F6M92', 'type', 'Male'),
            ('F9F141', 'hasParent', 'F9M139'), ('F7M125', 'type', 'Male'),
            ('F7M123', 'type', 'Brother'), ('F2M16', 'hasChild', 'F2F17'),
            ('F2M18', 'hasSibling', 'F2F17'), ('F6M78', 'hasChild', 'F6F79'),
            ('F10F195', 'type', 'Grandmother'), ('F4M59', 'type', 'Male'),
            ('F10F189', 'type', 'Person'), ('F6M92', 'hasSibling', 'F6F77'),
            ('F8M138', 'hasParent', 'F8M136'), ('F7M104', 'hasParent', 'F7M102'),
            ('F1M6', 'type', 'Person'), ('F6F72', 'married', 'F6M71'),
            ('F6M90', 'type', 'Thing'), ('F1F5', 'type', 'Sister'),
            ('F10M190', 'type', 'Male'), ('F6F89', 'type', 'Granddaughter'),
            ('F2M11', 'type', 'Brother'), ('F2M11', 'hasSibling', 'F2F26')
        ]
        output_dir = path or "KGs/Family"
        os.makedirs(output_dir, exist_ok=True)
        train_df = pd.DataFrame(relation_triples)
        train_df.to_csv(os.path.join(output_dir, "train.txt"), header=False, sep="\t", index=False)

    