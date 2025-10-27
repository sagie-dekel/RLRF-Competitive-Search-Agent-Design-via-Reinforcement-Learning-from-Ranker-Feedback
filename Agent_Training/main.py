import argparse
import json
import math
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM
from work.Agent_Training.reward_model import Reward_model
from work.Training_Data_Processing.load_data import get_data_loader_for_RLRF, get_data_loader_for_RM, load_data
from trl import PPOConfig, PPOTrainer, DPOConfig, DPOTrainer
from work.Agent_Training.ranker import Mean_Pooling_Ranker
from RLRF_main import RLRF
from huggingface_hub import login
import os
import itertools
import work.prompts as instructions


class GPTRewardLoss(nn.Module):
    """
    Define a loss function for comparison dataset approach in the reward model training
    """
    @staticmethod
    def forward(sw, sl, rank_winner=1, rank_loser=2):
        # Assumes sw and sl are both tensors of shape (batch_size,)
        return -torch.mean(torch.log(torch.sigmoid(abs(rank_winner - rank_loser) * (sw - sl))))


LOSS_FUNCTIONS = {
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "L1Loss": nn.L1Loss,
    "GPTRewardLoss": GPTRewardLoss
    # Add other loss functions here as needed
}

ACTIVATION_FUNCTIONS = {
    "Sigmoid": nn.Sigmoid,
    "ReLU": nn.ReLU
    # Add other activation functions here as needed
}


def get_epoch_based_lr_scheduler(optimizer, total_num_steps, warmup_ratio=0.1):
    """
    Create a learning rate scheduler for steps-based updates.
    :param optimizer: Optimizer instance
    :param total_num_steps: Total number of epochs
    :param warmup_ratio: Ratio of epochs for warmup
    :return: LambdaLR scheduler
    """
    # Number of warmup epochs
    warmup_steps = math.ceil(warmup_ratio * total_num_steps)

    # Define the lambda function for epoch-based schedule
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linearly increase learning rate during warmup
            return current_step / warmup_steps
        else:
            # Linearly decrease learning rate after warmup
            return max(0.0, (total_num_steps - current_step) / (total_num_steps - warmup_steps))

    # Create the scheduler
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def create_PPO_trainer(model_with_value_head, tokenizer, optimizer, HP_PPO: dict, ref_model_path=None,
                       train_dataset=None, **kwargs):
    """
    Create a PPO trainer for the RLRF method.
    Work with version 11.4 of trl.
    :param model_with_value_head: model to RLRF (with value head)
    :param tokenizer: tokenizer of the model
    :param optimizer: optimizer for the PPO trainer
    :param HP_PPO: dictionary containing hyperparameters
    :param ref_model_path: path to the reference model
    :param train_dataset: dataset for the trainer to use
    :return: PPO trainer for the RLRF (with no reward model)
    """
    tokenizer.pad_token = tokenizer.eos_token

    log_path = HP_PPO.pop("logging_dir", "./logs")
    model_save_path = HP_PPO.pop("model_save_path")
    num_layers_to_train_RLRF = HP_PPO.pop("num_layers_to_train_RLRF", 0)
    RLRF_model_config = HP_PPO.pop('model_config_kwargs_RLRF', {})

    # Create PPOConfig using parameters from HP_RLRF
    PPO_config = PPOConfig(
        reward_model="custom",
        mini_batch_size=HP_PPO.get("batch_size"),
        gradient_accumulation_steps=1,
        log_with='tensorboard',
        project_kwargs={"logging_dir": log_path},
        **HP_PPO
    )

    # Save hyperparameters to a JSON file
    save_hyperparameters(model_save_path, optimizer=optimizer, log_path=log_path, RLRF_model_config=RLRF_model_config,
                         num_layers_to_train_RLRF=num_layers_to_train_RLRF, **HP_PPO)

    # Create reference model:
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path,
                                                     torch_dtype=torch.bfloat16)

    parameter_names = [n for n, _ in ref_model.named_parameters()]
    # ref_model = deepcopy(model)
    # if no layers are shared, return copy of model
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    ref_model.eval()

    ppo_trainer = PPOTrainer(model=model_with_value_head, config=PPO_config, tokenizer=tokenizer, optimizer=optimizer,
                             ref_model=ref_model, dataset=train_dataset)
    return ppo_trainer


def create_DPO_trainer(model, tokenizer, optimizer, HP_DPO: dict, train_dataset: Dataset,
                       eval_dataset: Dataset = None, ref_model_path=None, **kwargs):
    """
    Create a DPO trainer for the RLRF method.
    Work with version 12.1.0 of trl.
    :param model: model to RLRF (with value head)
    :param tokenizer: tokenizer of the model
    :param optimizer: optimizer for the PPO trainer
    :param HP_DPO: hyperparameters for the DPO trainer
    :param train_dataset: dataset for the trainer to use
    :param accelerator: accelerator for distributed training
    :param eval_dataset: evaluation dataset
    :param ref_model_path: path to the reference model (optional)
    :param kwargs: additional keyword arguments
    :return: DPO trainer for the RLRF
    """
    tokenizer.pad_token = tokenizer.eos_token
    model_save_path = HP_DPO.pop("model_save_path")
    num_layers_to_train_RLRF = HP_DPO.pop("num_layers_to_train_RLRF", 0)
    RLRF_model_config = HP_DPO.pop('model_config_kwargs_RLRF', {})

    # Create PPOConfig using parameters from HP_RLRF
    DPO_config = DPOConfig(
        report_to='tensorboard',
        logging_first_step=True,
        output_dir=model_save_path,
        overwrite_output_dir=True,
        **HP_DPO
    )

    # Save hyperparameters to a JSON file
    save_hyperparameters(model_save_path, optimizer=optimizer, num_layers_to_train_RLRF=num_layers_to_train_RLRF,
                         RLRF_model_config=RLRF_model_config, **HP_DPO)

    """
    steps_number = (HP_DPO.get("num_train_epochs", 8) *
                    math.ceil(len(train_dataset) / HP_DPO.get("per_device_train_batch_size")))
    """
    # Assume deepspeed:
    steps_number = (HP_DPO.get("num_train_epochs", 8) *
                    math.ceil(len(train_dataset) / (HP_DPO.get("per_device_train_batch_size") * torch.cuda.device_count())))
    scheduler = get_epoch_based_lr_scheduler(optimizer, steps_number, warmup_ratio=0.1)

    # Create reference model:
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path,
                                                     torch_dtype=torch.bfloat16)

    parameter_names = [n for n, _ in ref_model.named_parameters()]
    # ref_model = deepcopy(model)
    # if no layers are shared, return copy of model
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    ref_model.eval()

    dpo_trainer = DPOTrainer(model=model, args=DPO_config, processing_class=tokenizer, ref_model=ref_model,
                             optimizers=(optimizer, scheduler), train_dataset=train_dataset, eval_dataset=eval_dataset)
    return dpo_trainer


def load_models_with_value_head(model_RLRF_name_or_path, RM_name_or_path, RLRF_model_config: dict = {},
                                base_RM_config: dict = {}, LLM_data_type="float32"):
    """
    Load the models for RLRF (main LLM with value head).
    :param model_RLRF_name_or_path: name or path of the model to align with RLRF
    :param RM_name_or_path: reward model load name repo or path
    :param RLRF_model_config: configuration for the RLRF model
    :param base_RM_config: configuration for the base model of the reward model
    :return:
    """
    # model for RLRF (need value head!):
    accelerator = Accelerator()
    if accelerator.state.deepspeed_plugin is None:
        model_RLRF = AutoModelForCausalLM.from_pretrained(model_RLRF_name_or_path, device_map="auto",
                                                          torch_dtype=getattr(torch, LLM_data_type))
    else:
        model_RLRF = AutoModelForCausalLM.from_pretrained(model_RLRF_name_or_path,
                                                          torch_dtype=getattr(torch, LLM_data_type))
    model_RLRF.pretrained_model.config.update(RLRF_model_config)
    tokenizer_RLRF = AutoTokenizer.from_pretrained(model_RLRF_name_or_path)
    tokenizer_RLRF.add_special_tokens({'pad_token': tokenizer_RLRF.eos_token})
    tokenizer_RLRF.padding_side = 'left'

    # Load reward model:
    RM_tokenizer = AutoTokenizer.from_pretrained(RM_name_or_path)
    if not os.path.isdir(RM_name_or_path):
        reward_model = AutoModelForMaskedLM.from_pretrained(RM_name_or_path, device_map="cuda",
                                                            **base_RM_config)
    else:
        reward_model = torch.load(f"{RM_name_or_path}/model.pt", weights_only=False).to('cuda')
    return model_RLRF, tokenizer_RLRF, reward_model, RM_tokenizer


def load_models(model_RLRF_name_or_path, RM_name_or_path, RLRF_model_config: dict = {},
                base_RM_config: dict = {}, LLM_data_type="float32"):
    """
    Load the models for RLRF.
    :param model_RLRF_name_or_path: name or path of the model to align with RLRF
    :param RM_name_or_path: reward model load name repo or path
    :param RLRF_model_config: configuration for the RLRF model
    :param base_RM_config: configuration for the base model of the reward model
    :return:
    """
    # model for RLRF:
    """
    model_RLRF = AutoModelForCausalLM.from_pretrained(model_RLRF_name_or_path, device_map="auto",
                                                      torch_dtype=getattr(torch, LLM_data_type))
    """
    model_RLRF = AutoModelForCausalLM.from_pretrained(model_RLRF_name_or_path,
                                                      torch_dtype=getattr(torch, LLM_data_type), device_map="auto")
    model_RLRF.config.update(RLRF_model_config)
    tokenizer_RLRF = AutoTokenizer.from_pretrained(model_RLRF_name_or_path)
    tokenizer_RLRF.add_special_tokens({'pad_token': tokenizer_RLRF.eos_token})
    tokenizer_RLRF.padding_side = 'left'

    # Load reward model:
    reward_model = None
    RM_tokenizer = None
    if RM_name_or_path is not None:
        RM_tokenizer = AutoTokenizer.from_pretrained(RM_name_or_path)
        if not os.path.isdir(RM_name_or_path):
            reward_model = AutoModel.from_pretrained(RM_name_or_path, device_map="cuda", **base_RM_config)
        else:
            reward_model = torch.load(f"{RM_name_or_path}/model.pt", weights_only=False).to('cuda')
    return model_RLRF, tokenizer_RLRF, reward_model, RM_tokenizer


def RLRF_Pipeline(queries_path, create_trainer_func, HP_RLRF, HP_RM, RM_train_data=None, value_head_needed=False,
                  add_classification_layer_to_reward_model=True, train_full_RM=False, train_RM=True,
                  RM_train_loss_func=nn.MSELoss(), RM_activation_function=None, relative_order=False,
                  rm_accelerator=None, create_perfernces_dataset=False, **kwargs):
    """
    Main pipeline for RLRF (Reinforcement Learning with Reward Function).

    Orchestrates the entire process, including model loading, dataset preparation, training configurations,
    and executing RLRF alignment.

    :param queries_path: Path to the queries file.
    :param create_trainer_func: Function to create the trainer for the RLRF model.
    :param HP_RLRF: Hyperparameters for the RLRF fine-tuning stage.
    :param HP_RM: Hyperparameters for reward model training.
    :param RM_train_data: Data for reward model training or its file path.
    :param value_head_needed: If True, loads models with a value head.
    :param add_classification_layer_to_reward_model: If True, adds a classification layer to the reward model.
    :param train_full_RM: If True, trains the full reward model.
    :param train_RM: If True, trains the reward model.
    :param RM_train_loss_func: Loss function for reward model training.
    :param RM_activation_function: Activation function for the reward model.
    :param relative_order: If True, dataset contains relative order (document pairs).
    :param rm_accelerator: Accelerator for reward model training.
    :param create_perfernces_dataset: If True, creates a preference dataset.
    :param kwargs: Additional keyword arguments.
    """
    model_RLRF_name_or_path, RM_model_name_or_path, RLRF_model_save_path = extract_model_paths(HP_RLRF, kwargs)

    # Load models
    model_RLRF, tokenizer_RLRF, reward_model, RM_tokenizer = load_models_pipeline(
        model_RLRF_name_or_path, RM_model_name_or_path, HP_RLRF, HP_RM, value_head_needed
    )

    # Set layers to train and optimizer
    set_layers_to_train(model_RLRF, HP_RLRF.get("num_layers_to_train_RLRF", 0), value_head_needed)
    optimizer_RLRF = optim.AdamW(filter(lambda param: param.requires_grad, model_RLRF.parameters()),
                                 lr=HP_RLRF.get('learning_rate', 1.41e-5))

    # Create datasets
    train_dataset_RLRF, eval_dataset = create_datasets(queries_path, tokenizer_RLRF, HP_RLRF, create_perfernces_dataset,
                                                       kwargs, model_RLRF, reward_model, RM_tokenizer)

    # Create trainer
    FT_trainer = create_trainer_func(model_RLRF, tokenizer_RLRF, optimizer_RLRF, HP_RLRF,
                                     train_dataset=train_dataset_RLRF, eval_dataset=eval_dataset,
                                     ref_model_path=model_RLRF_name_or_path)

    # Start RLRF process
    RLRF_manager = RLRF(FT_trainer)

    if train_RM:
        train_reward_model(add_classification_layer_to_reward_model, reward_model, RM_activation_function,
                           train_full_RM, RM_train_data, RM_tokenizer, relative_order, HP_RM, rm_accelerator,
                           RLRF_manager, RM_train_loss_func)
    else:
        RLRF_manager.set_reward_model(reward_model, RM_tokenizer, rm_accelerator)

    RLRF_manager.RLRF_DPO(model_save_path=RLRF_model_save_path)
    # RLRF_manager.RLRF_PPO(dataloader_RLRF, epochs=HP_RLRF.get('ppo_epochs', 12), model_save_path=RLRF_model_save_path)


def extract_model_paths(HP_RLRF, kwargs):
    """
    Extracts and validates model paths from hyperparameters and additional arguments.

    :param HP_RLRF: Dictionary containing hyperparameters for RLRF.
    :param kwargs: Additional keyword arguments containing model paths.
    :return: Tuple (model_RLRF_name_or_path, RM_model_name_or_path, RLRF_model_save_path).
    """
    model_RLRF_name_or_path = kwargs.get('model_RLRF_name_or_path')
    if model_RLRF_name_or_path is None:
        raise ValueError("Missing model name for 'model_RLRF_name_or_path'. Please provide a valid model name or path.")

    RM_model_name_or_path = kwargs.get('RM_name_or_path', None)
    RLRF_model_save_path = HP_RLRF.get('model_save_path', "./model")

    return model_RLRF_name_or_path, RM_model_name_or_path, RLRF_model_save_path


def load_models_pipeline(model_RLRF_name_or_path, RM_model_name_or_path, HP_RLRF, HP_RM, value_head_needed):
    """
    Loads models based on whether a value head is needed.

    :param model_RLRF_name_or_path: Path to the RLRF model.
    :param RM_model_name_or_path: Path to the reward model.
    :param HP_RLRF: Hyperparameters for the RLRF model.
    :param HP_RM: Hyperparameters for the reward model.
    :param value_head_needed: Boolean indicating if a value head is required.
    :return: Tuple (model_RLRF, tokenizer_RLRF, reward_model, RM_tokenizer).
    """
    if value_head_needed:
        return load_models_with_value_head(
            model_RLRF_name_or_path, RM_model_name_or_path,
            RLRF_model_config=HP_RLRF.get('model_config_kwargs_RLRF', {}),
            base_RM_config=HP_RM.get('base_model_config_kwargs_RM', {}),
            LLM_data_type=HP_RLRF.pop("torch_dtype", "float32")
        )
    else:
        return load_models(
            model_RLRF_name_or_path, RM_model_name_or_path,
            RLRF_model_config=HP_RLRF.get('model_config_kwargs_RLRF', {}),
            base_RM_config=HP_RM.get('base_model_config_kwargs_RM', {}),
            LLM_data_type=HP_RLRF.pop("torch_dtype", "float16")
        )


def create_datasets(queries_path, tokenizer_RLRF, HP_RLRF, create_perfernces_dataset, kwargs, model_RLRF, reward_model, RM_tokenizer):
    """
    Creates training and evaluation datasets.

    :param queries_path: Path to the queries file.
    :param tokenizer_RLRF: Tokenizer for the RLRF model.
    :param HP_RLRF: Hyperparameters for RLRF.
    :param create_perfernces_dataset: Boolean indicating whether to create a preference dataset.
    :param kwargs: Additional keyword arguments.
    :param model_RLRF: The RLRF model.
    :param reward_model: The reward model.
    :param RM_tokenizer: Tokenizer for the reward model.
    :return: Tuple (train_dataset_RLRF, eval_dataset).
    """
    queries = pd.read_csv(queries_path)
    N_sample = HP_RLRF.pop("N_sample", 5)
    eval_dataset_path = HP_RLRF.pop("eval_dataset_path", None)

    if create_perfernces_dataset:
        train_dataset_RLRF = kwargs.get("perfernces_dataset_path", None)
        if train_dataset_RLRF is None:
            if kwargs.get("use_baseline_doc_for_perfernces_dataset", False):
                train_dataset_RLRF = get_perfernces_dataset_from_ranker_and_baseline_doc(
                    queries, model_RLRF, tokenizer_RLRF, reward_model,
                    kwargs.get("baseline_document_path"), RM_tokenizer,
                    document_column=kwargs.get("baseline_doc_column", "rejected"),
                    N_sample=N_sample, save_path=os.path.dirname(queries_path)
                )
            else:
                train_dataset_RLRF = get_perfernces_dataset_from_ranker(
                    queries, model_RLRF, tokenizer_RLRF, reward_model, RM_tokenizer,
                    N_sample=N_sample, save_path=os.path.dirname(queries_path)
                )
        else:
            train_dataset_RLRF = pd.read_csv(train_dataset_RLRF)
            train_dataset_RLRF = Dataset.from_pandas(train_dataset_RLRF)
    else:
        train_dataset_RLRF = get_data_loader_for_RLRF(queries, tokenizer_RLRF,
                                                      batch_size=HP_RLRF.get("per_device_train_batch_size"))

    eval_dataset = pd.read_csv(eval_dataset_path) if eval_dataset_path else None
    if eval_dataset is not None:
        eval_dataset = Dataset.from_pandas(eval_dataset)

    return train_dataset_RLRF, eval_dataset


def train_reward_model(add_classification_layer_to_reward_model, reward_model, RM_activation_function, train_full_RM,
                       RM_train_data, RM_tokenizer, relative_order, HP_RM, rm_accelerator, RLRF_manager,
                       RM_train_loss_func):
    """
    Train a reward model with optional classification layer and specified training parameters.

    Parameters:
    add_classification_layer_to_reward_model (bool): Whether to add a classification layer to the reward model.
    reward_model (torch.nn.Module): The reward model to be trained.
    RM_activation_function (function): Activation function to be used in the reward model.
    train_full_RM (bool): Whether to train the entire reward model.
    RM_train_data (str or pd.DataFrame): Training data for the reward model, either as a file path or DataFrame.
    RM_tokenizer (Tokenizer): Tokenizer to be used for processing the training data.
    relative_order (bool): Whether to use relative ordering in training data.
    HP_RM (dict): Hyperparameters for training the reward model.
    rm_accelerator (Accelerator, optional): Accelerator for distributed training.
    RLRF_manager (RLRFManager): Manager for reward model training process.
    RM_train_loss_func (function): Loss function to be used for training the reward model.
    Returns:
    None
    """
    # Add classification layer to reward model if needed
    if add_classification_layer_to_reward_model:
        reward_model = Reward_model(reward_model, activation_function=RM_activation_function,
                                    train_full_model=train_full_RM).to('cuda')

    # Prepare dataloader for reward model training
    if isinstance(RM_train_data, str):
        dataloader_RM = get_data_loader_for_RM(pd.read_csv(RM_train_data), RM_tokenizer,
                                               relative_order=relative_order, batch_size=HP_RM.pop("batch_size"))
    else:
        dataloader_RM = get_data_loader_for_RM(RM_train_data, RM_tokenizer, relative_order=relative_order,
                                               batch_size=HP_RM.pop("batch_size"))

    # Set up the optimizer
    optimizer_RM = optim.AdamW(filter(lambda param: param.requires_grad, reward_model.parameters()),
                               lr=HP_RM.pop("learning_rate", 1e-6))

    # Prepare for distributed training if an accelerator is provided
    if rm_accelerator is not None:
        reward_model, optimizer_RM, dataloader_RM = rm_accelerator.prepare(reward_model, optimizer_RM, dataloader_RM)

    # Train the reward model
    RLRF_manager.train_RM(reward_model, RM_tokenizer, dataloader_RM, optimizer_RM, loss_func=RM_train_loss_func,
                          rm_accelerator=rm_accelerator, relative_order=relative_order, **HP_RM)


def get_perfernces_dataset(queries: pd.DataFrame, LLM, LLM_tokenizer, reward_model, reward_model_tokenizer, N_sample=5,
                           generation_kwargs: dict = None, save_path="/preference_dataset.csv"):
    """
    Create a preference dataset for queries
    :param queries: dataframe containing queries with column "query"
    :param LLM: LLM model for generation
    :param LLM_tokenizer: LLM tokenizer
    :param reward_model: Reward model for scoring
    :param reward_model_tokenizer: Reward model tokenizer
    :param N_sample: Number of samples to generate
    :param generation_kwargs: Generation kwargs for the LLM
    :param save_path: Path to save the output
    :return: Preference dataset as a Hugging Face Dataset, with "prompt", "chosen", and "rejected" columns
    """
    reward_model.eval()
    if generation_kwargs is None:
        generation_kwargs = {
            "pad_token_id": LLM_tokenizer.eos_token_id,
            "top_k": 0.0,
            "top_p": 0.9,
            "do_sample": True,
            "max_new_tokens": 250,
            "temperature": 0.8
        }
    # Load and prepare queries from CSV
    queries = queries['query'].tolist()

    # Placeholder for the dataset
    data = {"prompt": [], "chosen": [], "rejected": []}
    with torch.no_grad():
        # Loop through each query and create the preference data
        for query in tqdm(queries, desc="Processing Queries"):

            # Convert query to instruction format
            instruction = instructions.RLRF_MODEL_INSTRUCTION_without_prev_document.format(query)
            inputs = LLM_tokenizer(instruction, return_tensors="pt").to(LLM.device)
            # Generate N samples from LLM with specified temperature
            generated_samples = []
            for _ in range(N_sample):
                output = LLM.generate(**inputs, **generation_kwargs)
                generated_text = LLM_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                generated_samples.append(generated_text)

            # Score the generated samples with the reward model
            texts = [instructions.RM_INSTRUCTION_1_DOC.format(query, response) for response in
                     generated_samples]
            inputs = reward_model_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
                reward_model.device)

            with torch.inference_mode():
                rewards = reward_model(**inputs).squeeze(-1)

            # Sort the samples by rewards
            sorted_samples = sorted(zip(generated_samples, rewards), key=lambda x: x[1], reverse=True)

            # Select the highest and lowest scored samples for the dataset
            chosen_sample = sorted_samples[0][0]  # Highest scored sample
            rejected_sample = sorted_samples[-1][0]  # Lowest scored sample

            # Append to dataset
            data["prompt"].append(instruction)
            data["chosen"].append(chosen_sample)
            data["rejected"].append(rejected_sample)

    # Convert data dictionary to a Hugging Face Dataset
    preference_dataset = Dataset.from_dict(data)
    preference_df = preference_dataset.to_pandas()
    preference_df.to_csv(f"{save_path}/perfernces_dataset.csv", index=False)
    return preference_dataset


def get_perfernces_dataset_from_ranker(queries: pd.DataFrame, LLM, LLM_tokenizer, ranker_model, ranker_tokenizer,
                                       N_sample=5, generation_kwargs: dict = None,
                                       save_path="/preference_dataset.csv"):
    """
    Create a preference dataset for queries from ranker
    :param queries: dataframe containing queries with column "query"
    :param LLM: LLM model for generation
    :param LLM_tokenizer: LLM tokenizer
    :param ranker_model: Ranker for scoring
    :param ranker_tokenizer: Ranker tokenizer
    :param N_sample: Number of samples to generate
    :param generation_kwargs: Generation kwargs for the LLM
    :param save_path: Path to save the output
    :return: Preference dataset as a Hugging Face Dataset, with "prompt", "chosen", and "rejected" columns
    """
    ranker_model.eval()
    if generation_kwargs is None:
        generation_kwargs = {
            "pad_token_id": LLM_tokenizer.eos_token_id,
            "top_k": 0.0,
            "top_p": 0.9,
            "do_sample": True,
            "max_new_tokens": 250,
            "temperature": 0.8,
            "num_return_sequences": N_sample
        }
    ranker = Mean_Pooling_Ranker(ranker_model, ranker_tokenizer)

    # Load and prepare queries from CSV
    queries = queries['query'].tolist()

    # Placeholder for the dataset
    data = {"prompt": [], "chosen": [], "rejected": []}
    with torch.no_grad():
        # Loop through each query and create the preference data
        for query in tqdm(queries, desc="Processing Queries"):

            # Convert query to instruction format
            instruction = instructions.RLRF_MODEL_INSTRUCTION_without_prev_document.format(query)
            inputs = LLM_tokenizer(instruction, return_tensors="pt").to(LLM.device)
            # Generate N samples from LLM
            output = LLM.generate(**inputs, **generation_kwargs)
            # Decode the generated outputs
            generated_samples = [
                LLM_tokenizer.decode(output[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                for i in range(N_sample)
            ]

            # Score the generated samples with the ranker
            scores = ranker.get_scores_for_query(query, generated_samples)

            rank_samples_dic = [(doc, score) for doc, score in zip(generated_samples, scores)]

            # Sort the samples by rewards
            sorted_samples = sorted(rank_samples_dic, key=lambda x: x[1], reverse=True)

            # Select the highest and lowest scored samples for the dataset
            chosen_sample = sorted_samples[0][0]  # Highest scored sample
            rejected_sample = sorted_samples[-1][0]  # Lowest scored sample

            # Append to dataset
            data["prompt"].append(instruction)
            data["chosen"].append(chosen_sample)
            data["rejected"].append(rejected_sample)

    # Convert data dictionary to a Hugging Face Dataset
    preference_dataset = Dataset.from_dict(data)
    preference_df = preference_dataset.to_pandas()
    preference_df.to_csv(f"{save_path}/ranker_perfernces_dataset_no_baseline_doc.csv", index=False)
    return preference_dataset


def get_perfernces_dataset_from_ranker_and_baseline_doc(queries, LLM, LLM_tokenizer, ranker_model, baseline_documents,
                                                    ranker_tokenizer, document_column: str = "document", N_sample=5,
                                                    generation_kwargs: dict = None,
                                                    save_path="/preference_dataset.csv"):
    """
    Create a preference dataset for queries from ranker
    :param queries: dataframe or path to containing queries with column "query"
    :param LLM: LLM model for generation
    :param baseline_documents: dataframe or path to baseline documents dataframe
    :param LLM_tokenizer: LLM tokenizer
    :param document_column: column name of the documents in the baseline dataframe
    :param ranker_model: Ranker for scoring
    :param ranker_tokenizer: Ranker tokenizer
    :param N_sample: Number of samples to generate
    :param generation_kwargs: Generation kwargs for the LLM
    :param save_path: Path to save the output
    :return: Preference dataset as a Hugging Face Dataset, with "prompt", "chosen", and "rejected" columns
    """
    ranker_model.eval()
    if generation_kwargs is None:
        generation_kwargs = {
            "pad_token_id": LLM_tokenizer.eos_token_id,
            "top_k": 0.0,
            "top_p": 0.9,
            "do_sample": True,
            "max_new_tokens": 250,
            "temperature": 0.8,
            "num_return_sequences": N_sample
        }
    ranker = Mean_Pooling_Ranker(ranker_model, ranker_tokenizer)
    # Load and prepare queries and docs from CSV
    if isinstance(queries, str):
        queries = pd.read_csv(queries).tolist()
    else:
        queries = queries['query'].tolist()
    if isinstance(baseline_documents, str):
        documents = pd.read_csv(baseline_documents)[document_column].tolist()
    else:
        documents = baseline_documents[document_column].tolist()

    # Placeholder for the dataset
    data = {"prompt": [], "chosen": [], "rejected": []}
    with torch.no_grad():
        # Loop through each query and create the preference data
        for query, document in tqdm(zip(queries, documents), desc="Processing Queries"):

            # Convert query to instruction format
            instruction = instructions.RLRF_MODEL_INSTRUCTION_WITH_PREV_DOCUMENT.format(query, document)
            inputs = LLM_tokenizer(instruction, return_tensors="pt").to(LLM.device)
            # Generate N samples from LLM
            output = LLM.generate(**inputs, **generation_kwargs)
            # Decode the generated outputs
            generated_samples = [
                LLM_tokenizer.decode(output[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                for i in range(N_sample)
            ]

            # Add baseline doc in order to check for improvement :TODO check if necessary
            generated_samples.append(document)

            # Score the generated samples with the ranker
            scores = ranker.get_scores_for_query(query, generated_samples)

            rank_samples_dic = [(doc, score) for doc, score in zip(generated_samples, scores)]

            # Sort the samples by rewards
            sorted_samples = sorted(rank_samples_dic, key=lambda x: x[1], reverse=True)

            # Select the highest and lowest scored samples for the dataset
            chosen_sample = sorted_samples[0][0]  # Highest scored sample
            rejected_sample = sorted_samples[-1][0]  # Lowest scored sample

            # Append to dataset
            data["prompt"].append(instruction)
            data["chosen"].append(chosen_sample)
            data["rejected"].append(rejected_sample)

    # Convert data dictionary to a Hugging Face Dataset
    preference_dataset = Dataset.from_dict(data)
    preference_df = preference_dataset.to_pandas()
    preference_df.to_csv(f"{save_path}/ranker_perfernces_dataset_with_baseline_doc.csv", index=False)
    return preference_dataset


def set_layers_to_train(model, num_layers_to_train, does_value_head_exist):
    """
    Freeze all layers of the model except the last 'num_layers_to_train' layers (not include the value head).
    If the number of layers to train equals the total number of hidden layers, do not freeze any layers.
    :param model: A transformer model with value head (!) of the trainer to modify.
    :param num_layers_to_train: Number of layers to leave trainable (unfrozen).
    :param does_value_head_exist: True if the model has a value head, False otherwise.
    """
    # Get total number of transformer layers
    if does_value_head_exist:
        transformer_layers = model.pretrained_model.model.layers
        lm_head = model.pretrained_model.lm_head

    else:
        transformer_layers = model.model.layers
        lm_head = model.lm_head
    total_layers = len(transformer_layers)

    if num_layers_to_train == 'all':
        return

        # If the number of layers to train is less than the total layers, freeze
    if num_layers_to_train < total_layers:
        # Freeze all layers first
        for name, param in model.named_parameters():
            if "v_head" not in name:
                param.requires_grad = False

        # Unfreeze the last num_layers_to_train layers
        for layer in transformer_layers[-num_layers_to_train:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in lm_head.parameters():
            param.requires_grad = True


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


def get_unique_path(base_path, index=0):
    """Generates a unique directory name by adding an increasing index."""
    unique_path = f"{base_path}/{index}"

    while os.path.exists(unique_path):
        index += 1
        unique_path = f"{base_path}/{index}"

    return unique_path


def parse_json_arguments():
    """
    Parses the JSON configuration file provided via command-line arguments.

    :return: Dictionary containing parameters from the JSON file.
    """
    parser = argparse.ArgumentParser(description='Run RLRF Pipeline with parameters from a JSON file.')
    parser.add_argument('json_file', type=str, help='Path to the JSON configuration file')
    args = parser.parse_args()

    return load_params_from_json(args.json_file)


def authenticate_huggingface(params):
    """
    Authenticates with Hugging Face using the provided token.

    :param params: Dictionary containing configuration parameters.
    """
    token = params.get("token", "")
    if token:
        login(token=token, add_to_git_credential=True)


def load_loss_function(params):
    """
    Loads the specified loss function for the reward model training.

    :param params: Dictionary containing configuration parameters.
    :return: Loss function instance or None if not specified.
    """
    loss_function_name = params.get("HP_RM", {}).pop("RM_train_loss_func", None)
    if loss_function_name:
        return LOSS_FUNCTIONS[loss_function_name]()
    return None


def load_activation_function(params):
    """
    Loads the specified activation function for the reward model.

    :param params: Dictionary containing configuration parameters.
    :return: Activation function instance or None if not specified.
    """
    activation_function_name = params.pop("RM_activation_function", None)
    if activation_function_name:
        return ACTIVATION_FUNCTIONS[activation_function_name]()
    return None


def load_reward_model_data(params):
    """
    Loads reward model training data if required.

    :param params: Dictionary containing configuration parameters.
    :return: Reward model training data or None.
    """
    ranker_name = params.pop("ranker_model_name", None)
    if ranker_name is not None and params.get("train_RM", False):
        return load_and_score_data(**params)
    return params.pop("RM_train_data", None)


def generate_hp_combinations(params):
    """
    Generates all possible combinations of hyperparameters for HP_RLRF.

    :param params: Dictionary containing configuration parameters.
    :return: List of dictionaries containing hyperparameter combinations.
    """
    HP_RLRF_params = params.pop("HP_RLRF", {})
    keys, values = zip(*HP_RLRF_params.items())
    return [dict(zip(keys, combination)) for combination in itertools.product(*[v if isinstance(v, list) else [v] for v in values])]


def run_rlrf_pipeline(hp_combinations, params, RM_train_loss_func, RM_activation_function, rm_accelerator, RM_train_data):
    """
    Runs the RLRF pipeline for each combination of hyperparameters.

    :param hp_combinations: List of hyperparameter combinations for HP_RLRF.
    :param params: Dictionary containing other configuration parameters.
    :param RM_train_loss_func: Loss function for reward model training.
    :param RM_activation_function: Activation function for reward model.
    :param rm_accelerator: Accelerator instance for training.
    :param RM_train_data: Reward model training data.
    """
    for index, HP_RLRF_comb in enumerate(hp_combinations):
        torch.cuda.empty_cache()

        # Generate unique paths for model saving and logging
        HP_RLRF_comb["model_save_path"] = get_unique_path(HP_RLRF_comb.get('model_save_path'), index=index)
        HP_RLRF_comb["logging_dir"] = get_unique_path(HP_RLRF_comb.get('logging_dir'), index=index)

        print(f"Running RLRF pipeline with HP_RLRF: {HP_RLRF_comb}")

        # Run the RLRF pipeline
        RLRF_Pipeline(
            create_trainer_func=create_DPO_trainer,
            RM_train_loss_func=RM_train_loss_func,
            RM_activation_function=RM_activation_function,
            rm_accelerator=rm_accelerator,
            HP_RLRF=HP_RLRF_comb,
            HP_RM=params.pop("HP_RM", {}),
            RM_train_data=RM_train_data,
            **params
        )


def main():
    """
    Main function - runs the RLRF Pipeline with combinations of hyperparameters if any hyperparameter is a list.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize Accelerators for distributed training
    rm_accelerator = Accelerator()

    # Parse JSON file from command-line arguments
    params = parse_json_arguments()

    # Authenticate with Hugging Face if a token is provided
    authenticate_huggingface(params)

    # Load loss and activation functions
    RM_train_loss_func = load_loss_function(params)
    RM_activation_function = load_activation_function(params)

    # Load reward model training data if required
    RM_train_data = load_reward_model_data(params)

    # Generate hyperparameter combinations
    hp_combinations = generate_hp_combinations(params)

    # Run the RLRF pipeline for each hyperparameter combination
    run_rlrf_pipeline(hp_combinations, params, RM_train_loss_func, RM_activation_function, rm_accelerator,
                      RM_train_data)


if __name__ == '__main__':
    main()
