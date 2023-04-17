from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class JobTitleIdentifier:
    """
    A classifier that uses a fine-tuned BERT model to predict whether a given input text is a job title.
    """
    def __init__(self, model_path="bert-base-cased", weights_path="job_title_identifier.pt"):
        """
        Initializes the JobTitleClassifier with a pre-trained BERT model and custom weights.

        Args:
            model_path (str): The path of the pre-trained BERT model.
            weights_path (str): The path of the custom weights for the fine-tuned BERT model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self._FineTuneBERT(AutoModel.from_pretrained(model_path))
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def predict(self, input_text):
        """
        Predicts the class of the input text using the fine-tuned BERT model.

        Args:
            input_text (str): The input text to be classified.

        Returns:
            int: The predicted class of the input text (0 or 1).
        """
        token_ids = self.tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length=128)
        input_ids = torch.tensor([token_ids])
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)

        return predicted_class.item()

    class _FineTuneBERT(nn.Module):
        """
        A fine-tuned BERT model for job title classification.
        """
        def __init__(self, bert_model):
            """
            Initializes the fine-tuned BERT model with a pre-trained BERT model.

            Args:
                bert_model (transformers.PreTrainedModel): The pre-trained BERT model.
            """
            super().__init__()
            self.bert = bert_model
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(768, 2)

        def forward(self, input_ids, attention_mask):
            """
            The forward pass of the fine-tuned BERT model.

            Args:
                input_ids (torch.Tensor): The input token IDs.
                attention_mask (torch.Tensor): The attention mask.

            Returns:
                torch.Tensor: The logits of the predicted classes.
            """
            bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
            bert_output = bert_output[:, 0, :]
            bert_output = self.drop(bert_output)
            logits = self.out(bert_output)
            return logits
