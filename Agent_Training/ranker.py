import pandas as pd
import torch
import torch.nn.functional as F


class Mean_Pooling_Ranker:
    """
    E5 ranker class
    """
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer

    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform average pooling over the hidden states, masking out padding tokens.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def score(self, dataframe: pd.DataFrame, batch_size: int = 32, query_column_name='query',
              document_column_name='document'):
        """
        Score documents with respect to the query using the E5 model in batches.
        :param dataframe: A DataFrame containing 'query' and 'document' columns.
        :param batch_size: Number of rows to process in each batch.
        :param query_column_name: Name of the column containing the queries.
        :param document_column_name: Name of the column containing the documents.
        :return: The same DataFrame with a 'true_score' column.
        """
        true_scores = []

        # Process the DataFrame in batches
        for start_idx in range(0, len(dataframe), batch_size):
            end_idx = min(start_idx + batch_size, len(dataframe))
            batch_df = dataframe.iloc[start_idx:end_idx]

            all_input_texts = []
            for _, row in batch_df.iterrows():
                query = "query: " + str(row[query_column_name])
                document = "passage: " + str(row[document_column_name])
                all_input_texts.extend([query, document])
            print(all_input_texts)
            # Tokenize batch inputs
            inputs = self.tokenizer(all_input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            # Forward pass through the model
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Perform average pooling
            embeddings = Mean_Pooling_Ranker.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Compute cosine similarity scores in batches
            for i in range(0, len(embeddings), 2):
                query_embedding = embeddings[i]
                doc_embedding = embeddings[i + 1]
                score = (query_embedding @ doc_embedding.T).item()
                true_scores.append(score)

        dataframe['true_score'] = true_scores

    def score_and_set_winner(self, dataframe: pd.DataFrame, batch_size=32, query_column_name='query',
              document_1_column_name='document', document_2_column_name='document'):
        """
        Score document_1 and document_2 with respect to the query using the E5 model.
        The function returns the DataFrame with an additional 'winner' column,
        where '1' indicates document_1 wins, and '2' indicates document_2 wins.
        :param dataframe: A DataFrame containing 'query', 'document_1', and 'document_2' columns.
        :param batch_size: Batch size for processing the DataFrame.
        :param query_column_name: Name of the column containing the queries.
        :param document_1_column_name: Name of the column containing the first document.
        :param document_2_column_name: Name of the column containing the second document.
        :return: The same DataFrame with 'winner' column.
        """
        winners = []

        # Iterate through the DataFrame in batches
        for start_idx in range(0, len(dataframe), batch_size):
            end_idx = min(start_idx + batch_size, len(dataframe))
            batch_df = dataframe.iloc[start_idx:end_idx]

            all_input_texts = []
            for _, row in batch_df.iterrows():
                query = "query: " + row[query_column_name]
                document_1 = "passage: " + row[document_1_column_name]
                document_2 = "passage: " + row[document_2_column_name]

                all_input_texts.extend([query, document_1, document_2])

            # Tokenize the batch inputs
            inputs = self.tokenizer(all_input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")

            # Move input tensors to the model's device
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            # Forward pass through the model to get the embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Perform average pooling on the model's last hidden states
            embeddings = Mean_Pooling_Ranker.average_pool(outputs.last_hidden_state, inputs['attention_mask'])

            # Normalize the embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Process the embeddings in batches of 3 (query, document_1, document_2)
            for i in range(0, len(embeddings), 3):
                query_embedding = embeddings[i]
                doc_1_embedding = embeddings[i + 1]
                doc_2_embedding = embeddings[i + 2]

                # Compute the cosine similarity scores
                score_1 = (query_embedding @ doc_1_embedding.T).item()
                score_2 = (query_embedding @ doc_2_embedding.T).item()

                # Determine the winner
                winners.append(1 if score_1 > score_2 else 2)

        # Add winners to the DataFrame
        dataframe['winner'] = winners

    def get_scores_for_query(self, query: str, docs: list) -> list:
        """
        Get scores for a query and a list of documents.
        :param query: single query string
        :param docs: list of document strings
        :return: list of scores
        """
        all_input_texts = []
        for doc in docs:
            query = "query: " + query
            document = "passage: " + doc
            all_input_texts.extend([query, document])
        print(all_input_texts)
        # Tokenize the batch inputs
        inputs = self.tokenizer(all_input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")

        # Move input tensors to the model's device
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        # Forward pass through the model to get the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Perform average pooling on the model's last hidden states
        embeddings = Mean_Pooling_Ranker.average_pool(outputs.last_hidden_state, inputs['attention_mask'])

        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = []
        for i in range(0, len(embeddings), 2):
            query_embedding = embeddings[i]
            doc_embedding = embeddings[i + 1]

            # Compute the cosine similarity scores
            score = (query_embedding @ doc_embedding.T).item()

            # Determine the winner
            scores.append(score)

        return scores

    @property
    def parameters(self):
        return self.model.parameters



