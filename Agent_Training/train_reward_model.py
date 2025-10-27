import argparse
import json
import itertools
import pandas as pd
import torch
from accelerate import Accelerator
from torch import optim, nn, Tensor
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from evaluation.difference_RM_eval import reward_model_eval
#from evaluation.RM_eval import reward_model_eval
from work.Agent_Training.reward_model import Reward_model
from work.Training_Data_Processing.load_data import get_data_loader_for_RM, load_data, create_dataloader_DifferenceRM
from work.Agent_Training.ranker import Mean_Pooling_Ranker
from work.Agent_Training.RLRF_main import RLRF
from huggingface_hub import login
import os
import torch.nn.functional as F


class GPTRewardLoss(nn.Module):
    """
    Define a loss function for comparison dataset approach in the reward model training
    """
    @staticmethod
    def forward(sw: Tensor, sl: Tensor, rank_winner: Tensor = 1, rank_loser: Tensor = 2):
        # Determine element-wise if sw > sl, set difference to 1 if true, otherwise keep rank_loser - rank_winner
        difference = torch.where(sw > sl, torch.tensor(1.0, device=sw.device),
                                 torch.tensor(rank_loser - rank_winner, device=sw.device))

        # Compute the loss with the adjusted difference
        return -torch.mean(torch.log(torch.sigmoid(difference * (sw - sl))))
        #return -torch.mean(torch.log(torch.sigmoid((rank_loser - rank_winner) * (sw - sl))))


class DifferenceLoss(nn.Module):
    """
    Loss function for training a difference model.
    Computes the loss based on logits and the indicator function I(y1, y2).
    """

    def __init__(self):
        super(DifferenceLoss, self).__init__()

    @staticmethod
    def forward(score, self_score, target, rank_winner: Tensor = 1, rank_loser: Tensor = 2):
        """
        Forward pass to compute the difference loss with dynamic weights.

        Args:
            score (Tensor): Logits output by the model for predictions.
            self_score (Tensor): Self-comparison scores (e.g., regularization input).
            target (Tensor): Target binary values (0 or 1).
            rank_winner (Tensor): Tensor of ranks for the winner samples.
            rank_loser (Tensor): Tensor of ranks for the loser samples.

        Returns:
            Tensor: Computed loss value.
        """
        weight = rank_loser - rank_winner

        # Use BCEWithLogitsLoss to compute the loss
        loss = (F.binary_cross_entropy_with_logits(score, target.float(), weight=weight) +
                torch.tensor(0.1, device=self_score.device) * torch.mean(self_score ** 2))
        # :TODO: Add regularization term of opposite score to the loss
        return loss


class MarginRankingLoss(nn.Module):
    """
    Define a loss function for comparison dataset approach in the reward model training
    """
    @staticmethod
    def forward(sw, sl, rank_winner: torch.tensor, rank_loser: torch.tensor):
        return F.margin_ranking_loss(sw, sl, target=torch.sign(rank_loser - rank_winner), reduction='mean')


LOSS_FUNCTIONS = {
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "L1Loss": nn.L1Loss,
    "MarginRankingLoss": MarginRankingLoss,
    "GPTRewardLoss": GPTRewardLoss,
    "DifferenceLoss": DifferenceLoss
    # Add other loss functions here as needed
}

ACTIVATION_FUNCTIONS = {
    "Sigmoid": nn.Sigmoid,
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh
    # Add other activation functions here as needed
}


def load_models(RM_name_or_path, base_model_config_kwargs_RM: dict = {}):
    """
    Load the reward model
    :param RM_name_or_path: model name repo or path
    :param base_model_generation_kwargs_RM: base model of reward model generation kwargs
    :return:
    """
    # Load reward model:
    RM_tokenizer = AutoTokenizer.from_pretrained(RM_name_or_path)
    if not os.path.isdir(RM_name_or_path):
        reward_model = AutoModelForMaskedLM.from_pretrained(RM_name_or_path, device_map="cuda",
                                                            **base_model_config_kwargs_RM)
    else:
        reward_model = torch.load(f"{RM_name_or_path}/model.pt", weights_only=False).to('cuda')
    return reward_model, RM_tokenizer


def RM_training_optimize_HP(HP_RM, RM_train_data=None, add_classification_layer_to_reward_model=True,
                            train_full_RM=False, RM_train_loss_func=nn.MSELoss(), RM_activation_function=None,
                            relative_order=False, rm_accelerator=None, **kwargs):
    """
    optimize hyperparameters for the reward model training
    :param HP_RM: hyperparameters for the reward model training (if necessary)
    :param RM_train_data: data for reward model training or path to it (if necessary)
    :param add_classification_layer_to_reward_model: True if want to add classification layer to the reward model, False
    otherwise. if true a custom reward model class will be created.
    :param train_full_RM: True if want to train the full reward model, False otherwise
    :param RM_train_loss_func: loss function for the reward model training
    :param RM_activation_function: activation function for the reward model if needed
    :param relative_order: True if the data is in relative order (docs pairs), False otherwise
    :param rm_accelerator: accelerator for the reward model training (if necessary)
    :param kwargs: Additional keyword arguments.
        - RM_name_or_path: path or name of the reward model
    """
    RM_model_name_or_path = kwargs.get('RM_name_or_path')

    # Raise exceptions if any model name is missing
    if RM_model_name_or_path is None:
        raise ValueError("Missing model name for 'RM_base_model_name'. Please provide a valid model name.")

    # Load models:
    reward_model, RM_tokenizer = load_models(RM_model_name_or_path,
                                            base_model_config_kwargs_RM=HP_RM.get('base_model_config_kwargs_RM', {}))

    # Create custom reward model:
    if add_classification_layer_to_reward_model:
        reward_model = Reward_model(reward_model, activation_function=RM_activation_function,
                                    train_full_model=train_full_RM).to('cuda')

    # Start reward model training:
    RLRF_manager = RLRF(None)
    if isinstance(RM_train_data, str):
        dataloader_RM = create_dataloader_DifferenceRM(pd.read_csv(RM_train_data), RM_tokenizer,
                                                       batch_size=HP_RM.pop("batch_size"))
        """
        dataloader_RM = get_data_loader_for_RM(pd.read_csv(RM_train_data), RM_tokenizer,
                                               relative_order=relative_order, batch_size=HP_RM.pop("batch_size"))
        """
    else:
       dataloader_RM = get_data_loader_for_RM(RM_train_data, RM_tokenizer, relative_order=relative_order,
                                              batch_size=HP_RM.pop("batch_size"))
    optimizer_RM = optim.AdamW(filter(lambda param: param.requires_grad, reward_model.parameters()),
                               lr=HP_RM.pop("learning_rate", 1e-6))
    if rm_accelerator is not None:
        reward_model, optimizer_RM, dataloader_RM = rm_accelerator.prepare(reward_model, optimizer_RM,
                                                                           dataloader_RM)
    RLRF_manager.train_difference_RM(reward_model, RM_tokenizer, dataloader_RM, optimizer_RM, loss_func=RM_train_loss_func,
                                     rm_accelerator=rm_accelerator, **HP_RM)
    """
    RLRF_manager.train_RM(reward_model, RM_tokenizer, dataloader_RM, optimizer_RM, loss_func=RM_train_loss_func,
                          rm_accelerator=rm_accelerator, relative_order=relative_order, **HP_RM)
    """
    reward_model_eval(rm_accelerator.unwrap_model(reward_model), RM_tokenizer, RM_train_loss_func,
                      "/rg/kurland_prj/sagie.dekel/data/RLRF/small_data_set/scored_competition_history_eval.csv",
                      HP_RM.get("log_path"),
                      "/rg/kurland_prj/sagie.dekel/data/RLRF/small_data_set/queries.csv")
    """
    reward_model_eval(rm_accelerator.unwrap_model(reward_model), RM_tokenizer, RM_train_loss_func,
                      "/rg/kurland_prj/sagie.dekel/data/RLRF/small_data_set/scored_competition_history_eval.csv",
                      HP_RM.get("log_path"), True,
                      "/rg/kurland_prj/sagie.dekel/data/RLRF/small_data_set/queries.csv")
    """

def load_and_score_data(queries_path, data_path, ranker_model_name, relative_order=False):
    """
    Load the data, create a ranker, and score the documents.
    :param queries_path: Path to the queries file
    :param data_path: Path to the data file. the file contains docs and query id's
    :param ranker_model_name: Name of the ranker model
    :param relative_order: Whether to use relative ordering of documents
    :return: A DataFrame with scores and optionally, document comparisons
    """
    # Load ranker:
    ranker_tokenizer = AutoTokenizer.from_pretrained(ranker_model_name)
    ranker_model = AutoModel.from_pretrained(ranker_model_name, device_map="auto")

    # Load data
    docs_queries_data_frame = load_data(queries_path, data_path, "document_1", "query",
                                        "query_id", doc_2_column_name="document_2",
                                        relative_order=relative_order)

    # Create ranker and score documents
    ranker = Mean_Pooling_Ranker(ranker_model, ranker_tokenizer)
    if relative_order:
        ranker.score_and_set_winner(docs_queries_data_frame)
    else:
        ranker.score(docs_queries_data_frame)

    return docs_queries_data_frame


def load_params_from_json(json_file):
    """
    Load parameters from a JSON file.
    :param json_file: path to the JSON file.
    :return: parameters as a dictionary.
    """
    with open(json_file, 'r') as file:
        params = json.load(file)
    return params


def save_hyperparameters(save_path, **kwargs):
    # Save training hyperparameters as a JSON file
    hyperparameters = {key: str(value) if not isinstance(value, (str, int, float, bool, list, dict)) else value
                       for key, value in kwargs.items()}
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/training_hyperparameters.json", "w") as json_file:
        json.dump(hyperparameters, json_file, indent=4)


def main():
    """
    main function - activate reward model training with combinations of hyperparameters if any hyperparameter is a list.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize Accelerators for distributed training
    rm_accelerator = Accelerator()

    # Set up argument parser to accept the JSON file as a command-line argument
    parser = argparse.ArgumentParser(description='Run RLRF Pipeline with parameters from a JSON file.')
    parser.add_argument('json_file', type=str, help='Path to the JSON configuration file')
    args = parser.parse_args()

    # Load parameters from the provided JSON file
    params = load_params_from_json(args.json_file)

    # If exist token, log in to Hugging Face
    token = params.get("token", "")
    if token:
        login(token=token, add_to_git_credential=True)

    # Load loss function for the reward model:
    loss_function_name = params.get("HP_RM", None).pop("RM_train_loss_func", None)
    RM_train_loss_func = None
    if loss_function_name:
        RM_train_loss_func = LOSS_FUNCTIONS[loss_function_name]()

    # Load activation function for the reward model (if used):
    activation_function_name = params.pop("RM_activation_function", None)
    RM_activation_function = None
    if activation_function_name:
        RM_activation_function = ACTIVATION_FUNCTIONS[activation_function_name]()

    ranker_name = params.pop("ranker_model_name", None)
    if ranker_name is not None and params.get("train_RM", False):
        RM_train_data = load_and_score_data(**params)
    else:
        RM_train_data = params.pop("RM_train_data", None)

    # Get hyperparameters for RLRF and RM
    HP_RM = params.pop("HP_RM", {})

    # Find all hyperparameter combinations for HP_RM if any hyperparameter is a list
    keys, values = zip(*HP_RM.items())
    hyperparameter_combinations = [dict(zip(keys, combination)) for combination in
                                   itertools.product(*[v if isinstance(v, list) else [v] for v in values])]

    # Iterate through all combinations and run the RLRF pipeline
    for index, HP_RM_combination in enumerate(hyperparameter_combinations):
        torch.cuda.empty_cache()
        HP_RM_combination["model_save_path"] = f"{HP_RM_combination.get("model_save_path")}/{index}"
        HP_RM_combination["log_path"] = f"{HP_RM_combination.get("log_path")}/{index}"
        # Save hyperparameters to a JSON file
        save_hyperparameters(HP_RM_combination.get("model_save_path"), **HP_RM_combination)
        print(f"Running RLRF_Pipeline with HP_RM: {HP_RM_combination}")
        RM_training_optimize_HP(
            RM_train_loss_func=RM_train_loss_func,
            RM_activation_function=RM_activation_function,
            rm_accelerator=rm_accelerator,
            HP_RM=HP_RM_combination,
            RM_train_data=RM_train_data,
            **params
        )


if __name__ == '__main__':
    main()
