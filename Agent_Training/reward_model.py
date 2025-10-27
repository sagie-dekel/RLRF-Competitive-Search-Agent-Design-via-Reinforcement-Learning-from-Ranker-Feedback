import os
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class Reward_model(nn.Module, PyTorchModelHubMixin):
    """
    general reward model class with linear layer added to predict reward
    """
    def __init__(self, base_model, activation_function: nn.Module = None, train_full_model: bool = True):
        """
        :param base_model: base model for reward model. assuming that [CLS] token is added automatically by the model
        to the input sequence (at the beginning).
        :param activation_function: an activation function to the output (if needed)
        :param train_full_model: True will train the full model. otherwise just the linear layer will be trained.
        """
        super(Reward_model, self).__init__()
        # Use a pre-trained language model as a feature extractor
        self.base_model = base_model
        self.activation_function = activation_function
        # Freeze the base model parameters, so they are not trained:
        if not train_full_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        # A linear layer to output a scalar reward
        self.reward_head = nn.Linear(in_features=self.base_model.config.hidden_size, out_features=1,
                                     dtype=base_model.dtype)
        self.dtype = base_model.dtype

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Extract the hidden states from the pre-trained model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                  **kwargs)
        # Check for hidden states
        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden states
        else:
            raise ValueError("Base model doesn't provide hidden states, ensure output_hidden_states=True is set.")
        # Use the hidden states of the [CLS] token
        cls_embedding = hidden_states[:, 0, :]
        reward = self.reward_head(cls_embedding)
        if self.activation_function:
            reward = self.activation_function(reward)
        return reward

    @property
    def device(self):
        # Dynamically return the device of the first parameter
        return next(self.base_model.parameters()).device

    def save(self, save_path):
        """
        Saves the complete model to a specified path using torch.save().
        :param save_path: Path to save the model checkpoint.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the entire model as a checkpoint
        torch.save(self, save_path)
