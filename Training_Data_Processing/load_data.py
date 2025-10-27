import os
import random
import xml.etree.ElementTree as ET
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import work.prompts as instructions


def build_query_index(queries_path):
    """
    Function to create query index from a DataFrame.
    :param queries_path: path to the queries file with query and query_id columns.
    :return: queries index dictionary.
    """
    query_index = {}
    queries_dataframe = pd.read_csv(queries_path)

    # Loop through each row in the DataFrame and add query_id and query_text to the index
    for _, row in queries_dataframe.iterrows():
        query_id = row['query_id']
        query_text = row['query']

        # Add to index: query_id -> query
        query_index[query_id] = query_text

    return query_index


def create_dataframe_single_doc(csv_file_path, query_index, doc_column_name, query_column_name, query_id_column_name):
    """
    create data frame for exact score (no relative order between the documents)
    :param csv_file_path: path to a file that maps document to query id's
    :param query_index: index that maps query id's to query
    :param doc_column_name: doc column name
    :param query_column_name: query column name
    :param query_id_column_name: query id column name
    :return: a data frame with query-document pairs
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Map the query_id to the actual query text using the query_index
    df[query_column_name] = df[query_id_column_name].map(lambda x: query_index[x])
    # df = df.rename(columns={doc_column_name: 'document_1', query_column_name: 'query', query_id_column_name: 'query_id'}) # if you want the query id as well
    df = df.rename(columns={doc_column_name: 'document_1', query_column_name: 'query'})

    #return df[[doc_column_name, query_column_name, 'query_id']] # if you want the query id as well
    return df[[doc_column_name, query_column_name]]


def create_dataframe_2_docs(csv_file_path, query_index, doc_1_column_name, doc_2_column_name, query_column_name,
                            query_id_column_name):
    """
    create data frame for exact score (no relative order between the documents)
    :param csv_file_path: path to a file that maps document to query id's
    :param query_index: index that maps query id's to query
    :param doc_1_column_name: doc 1 column name
    :param doc_2_column_name: doc 2 column name
    :param query_column_name: query column name
    :param query_id_column_name: query id column name
    :return: a data frame with query-document pairs
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Map the query_id to the actual query text using the query_index
    df[query_column_name] = df[query_id_column_name].map(lambda x: query_index[x])
    # df = df.rename(columns={doc_1_column_name: 'document_1', doc_2_column_name: 'document_2', query_column_name:
    # 'query', query_id_column_name: 'query_id'}) # if you want the query id as well
    df = df.rename(columns={doc_1_column_name: 'document_1', doc_2_column_name: 'document_2', query_column_name:
        'query'})

    #return df[[doc_1_column_name, doc_2_column_name, query_column_name, query_id_column_name]] # if you want the query id as well
    return df[[doc_1_column_name, doc_2_column_name, query_column_name]]


class QueryDocumentDataset(Dataset):
    """
    Create QueryDocumentDataset for data loader of the reward model - return a query-document pair after tokenizing.
    Instructions format to the reward model also applied.
    The data frame needs to contain columns: document, query, and score (the score that was given to the doc with
    respect to the query).
    The output is input_ids and attention_mask to the reward model. The true_score also returned as a tensor with
    data type - torch.float32.
    """
    def __init__(self, dataframe, tokenizer, doc_column_name, score_column_name, query_column_name):
        self.tokenizer = tokenizer
        self.score_column_name = score_column_name

        # Tokenize all data at once
        self.tokenized_data = self.tokenize_data(dataframe, doc_column_name, query_column_name)

        # Store the true scores
        self.true_scores = torch.tensor(dataframe[score_column_name].values, dtype=torch.float32)

    def tokenize_data(self, dataframe, doc_column_name, query_column_name):
        # Format the instructions for all rows at once
        formatted_inputs = [
            instructions.RM_INSTRUCTION_1_DOC.format(row[query_column_name], row[doc_column_name])
            for _, row in dataframe.iterrows()
        ]

        # Tokenize all data at once, including padding
        input_tokens = self.tokenizer(
            formatted_inputs,
            padding=True,  # Pads all sequences to the same length
            truncation=True,  # Truncates if necessary
            return_tensors="pt"  # Return as PyTorch tensors
        )

        return input_tokens

    def __len__(self):
        return len(self.true_scores)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_data['input_ids'][idx],
            'attention_mask': self.tokenized_data['attention_mask'][idx],
            'true_score': self.true_scores[idx]
        }


class QueryDocumentDatasetForBERTModel(Dataset):
    """
    Same class as QueryDocumentDataset, just adjust for BERT model (add [SEP] token between query and document).
    Create QueryDocumentDataset for data loader of the reward model - return a query-document pair after tokenizing.
    Instructions format to the reward model also applied.
    The data frame needs to contain columns: document, query, and score (the score that was given to the doc with
    respect to the query).
    The output includes input_ids, attention_mask, and token_type_ids for BERT. The true_score is also returned as a
    tensor with data type - torch.float32.
    """
    def __init__(self, dataframe, tokenizer, doc_column_name, score_column_name, query_column_name):
        self.tokenizer = tokenizer
        self.doc_column_name = doc_column_name
        self.score_column_name = score_column_name
        self.query_column_name = query_column_name
        self.tokenized_data = self.tokenize_data(dataframe)

    def tokenize_data(self, dataframe):
        # Format the instructions for all rows at once
        formatted_inputs = [
            instructions.RM_MODEL_INSTRUCTION_1_DOC_FOR_BERT.format(row[self.query_column_name],
                                                                    row[self.doc_column_name])
            for _, row in dataframe.iterrows()
        ]

        # Tokenize all data at once, including padding
        input_tokens = self.tokenizer(
            formatted_inputs,
            padding=True,  # Pads all sequences to the same length
            truncation=True,  # Truncates if necessary
            return_tensors="pt"  # Return as PyTorch tensors
        )

        # Create and return tokenized data with scores
        tokenized_data = [
            {
                'input_ids': input_tokens['input_ids'][i],
                'attention_mask': input_tokens['attention_mask'][i],
                'true_score': torch.tensor(dataframe.iloc[i][self.score_column_name], dtype=torch.float32)
            }
            for i in range(len(dataframe))
        ]

        return tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]


class QueryDocumentPairDataset(Dataset):
    """
    Create QueryDocumentPairDataset for data loader - return two pairs of query-document with relative order (winner and
    loser) after tokenizing for the reward model.
    Instructions format to the reward model also applied.
    The dataframe needs to contain columns: document_1, document_2, query, winner (the better doc with respect to
    the query - 1 or 2).
    The output includes input_ids and attention_mask to the reward model (separated for winner and loser documents).
    """

    def __init__(self, dataframe, tokenizer, doc_1_column_name, doc_2_column_name, winner_column_name,
                 query_column_name, rank_1_column_name, rank_2_column_name):
        self.tokenizer = tokenizer

        # Tokenize all data at once and store it
        self.tokenized_data = self.tokenize_data(dataframe, doc_1_column_name, doc_2_column_name, query_column_name,
                                                 winner_column_name, rank_1_column_name, rank_2_column_name)

    def tokenize_data(self, dataframe, doc_1_column_name, doc_2_column_name, query_column_name,
                      winner_column_name, rank_1_column_name, rank_2_column_name):
        winner_input_texts = []
        loser_input_texts = []
        rank_winner_list, rank_loser_list = [], []
        for _, row in dataframe.iterrows():
            document_1 = row[doc_1_column_name]
            document_2 = row[doc_2_column_name]
            winner = row[winner_column_name]
            rank_1 = row[rank_1_column_name]
            rank_2 = row[rank_2_column_name]
            query = row[query_column_name]

            # Create texts for both winner and loser based on the value of the 'winner' column
            if winner == 1:
                winner_input_texts.append(instructions.RM_INSTRUCTION_1_DOC.format(query, document_1))
                loser_input_texts.append(instructions.RM_INSTRUCTION_1_DOC.format(query, document_2))
                rank_winner_list.append(int(rank_1))
                rank_loser_list.append(int(rank_2))
            else:
                winner_input_texts.append(instructions.RM_INSTRUCTION_1_DOC.format(query, document_2))
                loser_input_texts.append(instructions.RM_INSTRUCTION_1_DOC.format(query, document_1))
                rank_winner_list.append(int(rank_2))
                rank_loser_list.append(int(rank_1))

        # Tokenize all winner and loser texts at once
        winner_tokens = self.tokenizer(winner_input_texts, padding=True, truncation=True, return_tensors="pt")
        loser_tokens = self.tokenizer(loser_input_texts, padding=True, truncation=True, return_tensors="pt")

        return {
            'input_ids_winner': winner_tokens['input_ids'],
            'attention_mask_winner': winner_tokens['attention_mask'],
            'rank_winner': rank_winner_list,
            'input_ids_loser': loser_tokens['input_ids'],
            'attention_mask_loser': loser_tokens['attention_mask'],
            'rank_loser': rank_loser_list
        }

    def __len__(self):
        return len(self.tokenized_data['input_ids_winner'])

    def __getitem__(self, idx):
        return {
            'input_ids_winner': self.tokenized_data['input_ids_winner'][idx],
            'attention_mask_winner': self.tokenized_data['attention_mask_winner'][idx],
            'rank_winner': self.tokenized_data['rank_winner'][idx],
            'input_ids_loser': self.tokenized_data['input_ids_loser'][idx],
            'attention_mask_loser': self.tokenized_data['attention_mask_loser'][idx],
            'rank_loser': self.tokenized_data['rank_loser'][idx]
        }


class QueryDocumentsDifferenceRM(Dataset):
    """
    Create QueryDocumentsDifferenceRM for data loader - tokenize queries and docs for difference reward model training.
    Instructions format to the reward model also applied.
    The dataframe needs to contain columns: document_1, document_2, query, winner (the better doc with respect to
    the query - 1 or 2).
    """
    def __init__(self, dataframe, tokenizer, doc_1_column_name, doc_2_column_name, winner_column_name,
                 query_column_name, rank_1_column_name, rank_2_column_name):
        self.tokenizer = tokenizer

        # Tokenize all data at once and store it
        self.tokenized_data = self.tokenize_data(dataframe, doc_1_column_name, doc_2_column_name, query_column_name,
                                                 winner_column_name, rank_1_column_name, rank_2_column_name)

    def tokenize_data(self, dataframe, doc_1_column_name, doc_2_column_name, query_column_name,
                      winner_column_name, rank_1_column_name, rank_2_column_name):
        base_input_texts, self_input_text = [], []
        target = []
        rank_winner_list, rank_loser_list = [], []
        for _, row in dataframe.iterrows():
            document_1 = row[doc_1_column_name]
            document_2 = row[doc_2_column_name]
            winner = row[winner_column_name]
            rank_1 = row[rank_1_column_name]
            rank_2 = row[rank_2_column_name]
            query = row[query_column_name]

            coin_flip = random.random()

            if rank_1 == rank_2:
                pass

            # Create texts for both winner and loser based on the value of the 'winner' column
            if winner == 1:
                rank_winner_list.append(int(rank_1))
                rank_loser_list.append(int(rank_2))
                if coin_flip >= 0.5:
                    base_input_texts.append(instructions.Difference_RM_INSTRUCTION.format(query, document_1, document_2))
                    target.append(1)
                else:
                    base_input_texts.append(instructions.Difference_RM_INSTRUCTION.format(query, document_2, document_1))
                    target.append(-1)
            else:
                rank_winner_list.append(int(rank_2))
                rank_loser_list.append(int(rank_1))
                if coin_flip >= 0.5:
                    base_input_texts.append(instructions.Difference_RM_INSTRUCTION.format(query, document_2, document_1))
                    target.append(1)
                else:
                    base_input_texts.append(instructions.Difference_RM_INSTRUCTION.format(query, document_1, document_2))
                    target.append(-1)

            if random.random() >= 0.5:
                self_input_text.append(instructions.RM_INSTRUCTION_1_DOC.format(query, document_1, document_1))
            else:
                self_input_text.append(instructions.Difference_RM_INSTRUCTION.format(query, document_2, document_2))

        # Tokenize all winner and loser texts at once
        base_input = self.tokenizer(base_input_texts, padding=True, truncation=True, return_tensors="pt")
        self_input = self.tokenizer(self_input_text, padding=True, truncation=True, return_tensors="pt")

        return {
            'base_input': base_input,
            'self_input': self_input,
            'target': torch.tensor(target, dtype=torch.float32),
            'rank_winner': torch.tensor(rank_winner_list, dtype=torch.float32),
            'rank_loser': torch.tensor(rank_loser_list, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.tokenized_data['base_input']['input_ids'])

    def __getitem__(self, idx):
        return {
            'base_input': {key: val[idx] for key, val in self.tokenized_data['base_input'].items()},
            'self_input': {key: val[idx] for key, val in self.tokenized_data['self_input'].items()},
            'target': self.tokenized_data['target'][idx],
            'rank_winner': self.tokenized_data['rank_winner'][idx],
            'rank_loser': self.tokenized_data['rank_loser'][idx],
        }


class QueryInstructionDataset(Dataset):
    """
    create QueryInstructionDataset for data loader - return a query-document pair after tokenizing for the RLRF model.
    instructions format to the reward model also applied (not include the previous document inside).
    the data frame need to contain query column.
    The output is input_ids and attention_mask to the RLRF model.
    """
    def __init__(self, dataframe, tokenizer, query_column_name):
        self.tokenizer = tokenizer
        self.query_column_name = query_column_name
        self.tokenized_data = self.tokenize_data(dataframe)

    def tokenize_data(self, dataframe):
        tokenized_data = []
        for idx, row in dataframe.iterrows():
            query = row[self.query_column_name]
            input_tokens = self.tokenizer(
                instructions.RLRF_MODEL_INSTRUCTION_without_prev_document.format(query),
                padding=False, truncation=True, return_tensors="pt")
            # Store the tokenized input ids and attention mask
            tokenized_data.append({
                'input_ids': input_tokens['input_ids'].squeeze(0),
                'attention_mask': input_tokens['attention_mask'].squeeze(0),
                'query': query
            })
        return tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]


def custom_collate_for_RLRF(batch):
    # custom function to data loader in order to load queries from different lengths in the same batch
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    queries = [item['query'] for item in batch]  # This remains as a list, which does not need stacking

    # Pad input_ids and attention_masks (delete comment if needed)
    #padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    #padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'query': queries  # Return as list of strings
    }


def create_dataloader_RM_single_doc(dataframe, tokenizer, batch_size=16):
    """
    Create dataloader for the reward model where there is only a score to a document with respect to query (no relative
    order).
    :param dataframe: the data frame need to contain columns: document, query, score (the score that was given to the
    doc with respect to the query).
    :param tokenizer:
    :param batch_size:
    :return: torch.utils.data.DataLoader for reward model training
    """
    if "bert" in tokenizer.name_or_path.lower():
        dataset = QueryDocumentDatasetForBERTModel(dataframe, tokenizer, "document",
                                                   "true_score", "query")
    else:
        dataset = QueryDocumentDataset(dataframe, tokenizer, "document", "true_score",
                                       "query")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_dataloader_RM_relative_order(dataframe, tokenizer, batch_size=16):
    """
    Create dataloader for the reward model where there is relative order between the documents.
    :param dataframe: the data frame need to contain columns: document_1, document_2, query, winner (the better doc with
    respect to the query - 1 or 2).
    :param tokenizer: tokenizer of the reward model
    :param batch_size: batch size for the data loader
    :return: torch.utils.data.DataLoader for reward model training
    """
    dataset = QueryDocumentPairDataset(dataframe, tokenizer, "document_1", "document_2",
                                       "winner", "query", "rank_1", "rank_2")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_dataloader_DifferenceRM(dataframe, tokenizer, batch_size=16):
    """
    Create dataloader for difference reward model.
    :param dataframe: the data frame need to contain columns: document_1, document_2, query, winner (the better doc with
    respect to the query - 1 or 2).
    :param tokenizer: tokenizer of the reward model
    :param batch_size: batch size for the data loader
    :return: torch.utils.data.DataLoader for reward model training
    """
    dataset = QueryDocumentsDifferenceRM(dataframe, tokenizer, "document_1", "document_2",
                                         "winner", "query", "rank_1", "rank_2")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_dataloader_RLRF(dataframe, tokenizer, batch_size=16):
    """
    Create dataloader for the RLRF model.
    :param dataframe: need to contain query column.
    :param tokenizer:
    :param batch_size:
    :return: torch.utils.data.DataLoader for RLRF model training
    """
    dataset = QueryInstructionDataset(dataframe, tokenizer, "query")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_for_RLRF,
                            drop_last=True)
    return dataloader


def load_data(queries_path, docs_path, doc_1_column_name, query_column_name, query_id_column_name,
              doc_2_column_name="", relative_order=True):
    """
    load data where the queries in xml files and docs for query are in csv
    :param queries_path: queries csv file path
    :param docs_path: docs and query id's csv file path
    :param doc_1_column_name:
    :param query_column_name: query id's column name
    :param query_id_column_name: query id's column name
    :param doc_2_column_name: document 2 column name
    :param relative_order: if true load data frame with pairs of documents with respect to a query. otherwise load data
    frame with 1 document with respect to a query.
    :return: data frame with columns: query, document_1, document_2 (if relative_order is true)
    """
    # Build the query index from XML files
    query_index = build_query_index(queries_path)
    # Create a DataFrame from the CSV file and the query index
    if relative_order:
        dataframe = create_dataframe_2_docs(docs_path, query_index, doc_1_column_name, doc_2_column_name,
                                            query_column_name, query_id_column_name)
    else:
        dataframe = create_dataframe_single_doc(docs_path, query_index, doc_1_column_name, query_column_name,
                                                query_id_column_name)
    return dataframe


def get_data_loader_for_RM(dataframe, tokenizer_RM, relative_order=False, batch_size=16):
    # give score to each query-document pair by the ranker:
    if relative_order:
        dataloader_RM = create_dataloader_RM_relative_order(dataframe, tokenizer_RM, batch_size)
    else:
        dataloader_RM = create_dataloader_RM_single_doc(dataframe, tokenizer_RM, batch_size)
    return dataloader_RM


def get_data_loader_for_RLRF(dataframe, tokenizer_PPO, batch_size=16):
    dataloader_PPO = create_dataloader_RLRF(dataframe, tokenizer_PPO, batch_size)
    return dataloader_PPO







