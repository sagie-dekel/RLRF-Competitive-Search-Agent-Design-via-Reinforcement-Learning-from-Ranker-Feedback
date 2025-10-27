import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import work.prompts as instructions
from deepspeed.accelerator import get_accelerator


class RLRF:
    """
    Reinforcement Learning manager class.
    Reward model training or RL algorithms performed here using trl Trainer.
    Interfacing with accelerator or deepspeed for distributed training or mixed precision (only for DPO for now).
    """
    def __init__(self, FTTrainer):
        """
        :param FTTrainer: (Fine-Tune Trainer) trainer to use in the RLRF
        """
        self.FTTrainer = FTTrainer
        # Set attributes of reward model for future use:
        self.RM = None
        self.RM_tokenizer = None
        self.rm_accelerator = None
        self.push_to_hub_path = ""

    def set_reward_model(self, RM, RM_tokenizer, rm_accelerator):
        """
        set the reward model and the tokenizer
        :param RM: reward model for RL. assuming the reward model needs attention mask!
        :param RM_tokenizer: tokenizer of the reward model
        """
        self.RM = RM
        self.RM_tokenizer = RM_tokenizer
        self.rm_accelerator = rm_accelerator

    def train_RM(self, SFT_RM, RM_tokenizer, dataloader: torch.utils.data.DataLoader, optimizer,
                 relative_order=False, epochs=16, loss_func=nn.MSELoss(), rm_accelerator=None,
                 model_save_path="./models/RLRF_model", log_path="./logs/reward_model", **kwargs):
        """
        main function to train the reward model
        :param SFT_RM: base model to train. assuming the reward model needs attention mask!
        :param RM_tokenizer: tokenizer of the reward model
        :param dataloader: the data in torch.utils.data.DataLoader format! the batch needs to have input_ids and
        attention mask. the data needs to be already tokenized!
        :param epochs: number of epochs to commit
        :param optimizer: optimizer to use
        :param relative_order: does the data is pairs with winner and losser (relative order) or a document and his
        true score
        :param loss_func: loss function to use. must match to the relative_order! (if pairs of documents and the order
        then the loss compute respectively for the difference between winner and losser score, otherwise compute
        respectively to the difference between the score and true score)
        :param rm_accelerator: accelerator to use in the training (if is not None assume the mode, optimizer and data
        loader are already prepared)
        :param model_save_path: path to save the final model
        :param log_path: path to save the loss data (Average Loss per Epoch)
        :param kwargs: additional arguments for RM training
        """
        if self.RM:
            raise ValueError("RM already exists")

        writer = SummaryWriter(log_dir=log_path)
        # start training loop:
        SFT_RM.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct_predictions = 0
            total_comparisons = 0
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}, batch: "):
                # if we need to tokenize the input:
                # inputs = tokenizer(input_tokens, padding=True, truncation=True, return_tensors="pt")
                if relative_order:
                    batch_loss, batch_correct = RLRF.process_batch_RM_relative_order(SFT_RM, batch, loss_func)
                    correct_predictions += batch_correct
                    total_comparisons += len(batch["input_ids_winner"])
                else:
                    batch_loss = RLRF.process_batch_RM_no_relative_order(SFT_RM, batch, loss_func)
                # calculate loss and make step:
                if rm_accelerator is None:
                    total_loss = RLRF.optimize_RM_batch(optimizer, batch_loss, total_loss)
                else:
                    total_loss = RLRF.optimize_RM_batch_with_accelerator(optimizer, batch_loss, total_loss,
                                                                         rm_accelerator)
            # Average batch loss for the epoch
            avg_loss = total_loss / len(dataloader)
            epoch_accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0.0

            writer.add_scalar('Loss per Epoch', avg_loss, epoch)
            writer.add_scalar('Accuracy per Epoch', epoch_accuracy, epoch)

        writer.close()
        self.RM = SFT_RM
        self.rm_accelerator = rm_accelerator
        self.RM_tokenizer = RM_tokenizer

        # Save model:
        rm_accelerator.wait_for_everyone()
        unwrapped_model = rm_accelerator.unwrap_model(SFT_RM)
        unwrapped_model.save(f"{model_save_path}/model.pt")
        RM_tokenizer.save_pretrained(model_save_path)

    def train_difference_RM(self, SFT_RM, RM_tokenizer, dataloader: torch.utils.data.DataLoader, optimizer,
                 epochs=16, loss_func=nn.MSELoss(), rm_accelerator=None,
                 model_save_path="./models/RLRF_model", log_path="./logs/reward_model", **kwargs):
        """
        main function to train the difference reward model
        :param SFT_RM: base model to train. assuming the reward model needs attention mask!
        :param RM_tokenizer: tokenizer of the reward model
        :param dataloader: the data in torch.utils.data.DataLoader format! the batch needs to have input_ids and
        attention mask. the data needs to be already tokenized!
        :param epochs: number of epochs to commit
        :param optimizer: optimizer to use
        :param relative_order: does the data is pairs with winner and losser (relative order) or a document and his
        true score
        :param loss_func: loss function to use. must match to the relative_order! (if pairs of documents and the order
        then the loss compute respectively for the difference between winner and losser score, otherwise compute
        respectively to the difference between the score and true score)
        :param rm_accelerator: accelerator to use in the training (if is not None assume the mode, optimizer and data
        loader are already prepared)
        :param model_save_path: path to save the final model
        :param log_path: path to save the loss data (Average Loss per Epoch)
        :param kwargs: additional arguments for RM training
        """
        if self.RM:
            raise ValueError("RM already exists")

        writer = SummaryWriter(log_dir=log_path)
        # start training loop:
        SFT_RM.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct_predictions = 0
            total_comparisons = 0
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}, batch: "):
                batch_loss, batch_correct = RLRF.process_batch_RM_Difference_model(SFT_RM, batch, loss_func)
                correct_predictions += batch_correct
                total_comparisons += len(batch["base_input"]['input_ids'])
                # calculate loss and make step:
                if rm_accelerator is None:
                    total_loss = RLRF.optimize_RM_batch(optimizer, batch_loss, total_loss)
                else:
                    total_loss = RLRF.optimize_RM_batch_with_accelerator(optimizer, batch_loss, total_loss,
                                                                         rm_accelerator)
            # Average batch loss for the epoch
            avg_loss = total_loss / len(dataloader)
            epoch_accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0.0

            writer.add_scalar('Loss per Epoch', avg_loss, epoch)
            writer.add_scalar('Accuracy per Epoch', epoch_accuracy, epoch)

        writer.close()
        self.RM = SFT_RM
        self.rm_accelerator = rm_accelerator
        self.RM_tokenizer = RM_tokenizer

        # Save model:
        rm_accelerator.wait_for_everyone()
        unwrapped_model = rm_accelerator.unwrap_model(SFT_RM)
        unwrapped_model.save(f"{model_save_path}/model.pt")
        RM_tokenizer.save_pretrained(model_save_path)

    @staticmethod
    def process_batch_RM_no_relative_order(SFT_RM, batch, loss_func):
        """
        generate reward for query-document pair in the batch and calculate loss
        :param SFT_RM: Reward model
        :param batch: batch to process
        :param loss_func: loss function to use
        :return: the final loss of the batch
        """
        true_score = batch.pop("true_score")
        predicted_reward = SFT_RM(**batch).squeeze(-1)
        # Compute loss
        batch_loss = loss_func(predicted_reward.to(torch.float32), true_score.to(torch.float32))
        print(f"batch_loss: {batch_loss}, predicted_reward: {predicted_reward}, true_score: {true_score}")

        return batch_loss

    @staticmethod
    def process_batch_RM_relative_order(SFT_RM, batch, loss_func):
        """
        generate reward for query-document (winner and loser) pairs in the batch and calculate loss
        :param SFT_RM: Reward model
        :param batch: batch to process
        :param loss_func: loss function to use
        :return: the final loss of the batch
        """
        rank_winner = batch.pop("rank_winner")
        rank_loser = batch.pop("rank_loser")
        # Load the input tokens and attention mask for both winning and losing documents
        input_ids_winner = batch["input_ids_winner"]
        attention_mask_winner = batch["attention_mask_winner"]
        input_ids_loser = batch["input_ids_loser"]
        attention_mask_loser = batch["attention_mask_loser"]

        # Forward pass through the model for both winners and losers in one go
        reward_winner = SFT_RM(input_ids_winner, attention_mask=attention_mask_winner).squeeze(-1)
        reward_loser = SFT_RM(input_ids_loser, attention_mask=attention_mask_loser).squeeze(-1)

        # Compute the loss for the entire batch (winner vs loser)
        batch_loss = loss_func(reward_winner.to(torch.float32), reward_loser.to(torch.float32), rank_winner=rank_winner,
                               rank_loser=rank_loser)

        # Calculate accuracy: count where winner's score > loser's score
        correct_predictions = (reward_winner > reward_loser).sum().item()

        print(f"batch_loss: {batch_loss}, reward_winner: {reward_winner}, reward_loser: {reward_loser}")

        return batch_loss, correct_predictions

    @staticmethod
    def process_batch_RM_Difference_model(SFT_RM, batch, loss_func):
        """
        generate reward for query-document-document for difference reward mdoel
        :param SFT_RM: Reward model
        :param batch: batch to process
        :param loss_func: loss function to use
        :return: the final loss of the batch
        """
        rank_winner = batch["rank_winner"]
        rank_loser = batch["rank_loser"]
        target = batch["target"]
        base_input = batch["base_input"]
        self_input = batch["self_input"]

        # Forward pass through the model for both winners and losers in one go
        score = SFT_RM(**base_input).squeeze(-1)
        self_score = SFT_RM(**self_input).squeeze(-1)

        # Compute the loss for the entire batch (winner vs loser)
        batch_loss = loss_func(score, self_score, target, rank_winner=rank_winner, rank_loser=rank_loser)

        # Calculate accuracy:
        correct_predictions = torch.where(target == 1.0, score > 0, score < 0).sum().item()

        print(f"batch_loss: {batch_loss}, correct_predictions: {correct_predictions}")

        return batch_loss, correct_predictions

    @staticmethod
    def optimize_RM_batch(optimizer, loss, total_loss):
        """
        Make gradient step and update cumulative loss
        :param optimizer: optimizer to use
        :param loss: loss of the current batch
        :param total_loss: total loss of the epoch up until current batch
        :return: the cumulative loss of the epoch
        """
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accumulate loss
        total_loss += loss.item()

        return total_loss

    @staticmethod
    def optimize_RM_batch_with_accelerator(optimizer, loss, total_loss, accelerator):
        """
        Make gradient step and update cumulative loss
        :param optimizer: optimizer to use
        :param loss: loss of the current batch
        :param total_loss: total loss of the epoch up until current batch
        :param accelerator: accelerator to use
        :return: the cumulative loss of the epoch
        """
        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # Accumulate loss
        total_loss += loss.item()

        return total_loss

    def RLRF_PPO(self, dataloader: torch.utils.data.DataLoader, epochs=16, model_save_path="./models/RLRF_model"):
        """
        Preforming RLRF using the given reward model and the policy trainer.
        :param dataloader: loader of the training data after tokenizing! It should be a torch.utils.data.DataLoader.
        the batch needs to have input_ids and attention mask.
        already include instruction to the model inside.
        :param epochs: number of epochs to commit
        :param model_save_path: path to save the final model
        """
        start_time = time.time()  # Record the start time
        max_duration = 20 * 60 * 60  # 20 hours in seconds
        dataloader = self.FTTrainer.accelerator.prepare(dataloader)

        # recommended attributes from TRL to prevent negative KL score:
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1,
            "do_sample": True,
            "max_new_tokens": 250,
            "pad_token_id": self.FTTrainer.tokenizer.eos_token_id,
            "temperature": 0.7
        }

        # Start RLRF training loop:
        for epoch in range(1, epochs + 1):
            for batch in tqdm(dataloader, desc=f"Epoch {epoch}, batch: "):
                # Process each batch
                self.process_batch_RLRF_PPO(batch, generation_kwargs)

                # Check elapsed time
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_duration:
                    print(f"Stopping training early at epoch {epoch} due to 20-hour limit.")
                    self.FTTrainer.accelerator.unwrap_model(self.FTTrainer.model).push_to_hub(self.push_to_hub_path,
                                                                                              private=True)
                    self.FTTrainer.tokenizer.push_to_hub(self.push_to_hub_path, private=True)

                    return
        # Save model
        self.FTTrainer.accelerator.unwrap_model(self.FTTrainer.model).push_to_hub(self.push_to_hub_path, private=True)
        self.FTTrainer.tokenizer.push_to_hub(self.push_to_hub_path, private=True)
        # In order to save model, tokenizer and model card locally:
        self.FTTrainer.accelerator.unwrap_model(self.FTTrainer.model).save_pretrained(model_save_path)
        self.FTTrainer.tokenizer.save_pretrained(model_save_path)

    def process_batch_RLRF_PPO(self, batch, generation_kwargs: dict):
        """
        Process batch of RLRF
        :param batch: the batch to process
        :param generation_kwargs: arguments for generation of the alignment model
        """
        input_ids_list = [input_tensor for input_tensor in batch["input_ids"]]
        # Get response from SFTModel
        response_tensors = self.FTTrainer.generate(input_ids_list, batch_size=len(input_ids_list), return_prompt=False,
                                                   **generation_kwargs)
        batch["response"] = [self.FTTrainer.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in
                             response_tensors]
        print(f"response: {batch["response"]}")

        # Compute reward score
        texts = [instructions.RM_INSTRUCTION_1_DOC.format(q, r)
                 for q, r in zip(batch["query"], batch["response"])]
        # Tokenize the input texts
        inputs = self.RM_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Ensure inputs are moved to the appropriate device if rm_accelerator is available
        if self.rm_accelerator is not None:
            inputs = {k: v.to(self.rm_accelerator.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.RM.device) for k, v in inputs.items()}

        # Pass the tokenized inputs through the reward model
        with torch.inference_mode():
            rewards = self.RM(**inputs)
        rewards = rewards.squeeze(-1)  # Remove the last dimension (batch size)
        rewards = [reward.to(self.FTTrainer.accelerator.device, dtype=torch.float32) for reward in rewards]

        print(f"reward for batch: {rewards}")

        # Run a Trainer step
        stats = self.FTTrainer.step(input_ids_list, response_tensors, rewards)

        print(f"Mean Reward ppo: {stats.get('ppo/mean_scores', None)}")
        print(f"Average Loss: {stats.get('ppo/loss/total', None)}")
        print(f"Mean returns: {stats.get('ppo/returns/mean', None)}")

        self.FTTrainer.log_stats(stats, batch, rewards)

    def RLRF_DPO(self, model_save_path="./models/RLRF_model"):
        """
        Preforming RLRF using DPO algorithm
        :param model_save_path: path to save the final model
        """
        self.FTTrainer.train()
        get_accelerator().synchronize()
        # Save model
        if self.FTTrainer.is_deepspeed_enabled:
            self.FTTrainer.save_model(model_save_path)
        else:
            # Push to hub:
            #self.FTTrainer.accelerator.unwrap_model(self.FTTrainer.model).push_to_hub(self.push_to_hub_path, private=True)
            #self.FTTrainer.processing_class.push_to_hub(self.push_to_hub_path, private=True)
            self.FTTrainer.accelerator.unwrap_model(self.FTTrainer.model).save_pretrained(model_save_path)
            self.FTTrainer.processing_class.save_pretrained(model_save_path)
            self.FTTrainer.create_model_card(model_name="RLRF_DPO")

