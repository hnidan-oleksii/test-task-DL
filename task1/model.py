import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset
from tqdm.auto import tqdm
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistilBertMountainTokenClassifier:
    def __init__(
            self,
            model_name='distilbert-base-cased',
            num_labels=3,
            log_dir='./runs',
            max_length=512
    ):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        self.model = DistilBertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        self.loss_fn = CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.label2id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.max_length = max_length

    def train(
            self,
            train_set_path,
            val_set_path,
            batch_size=16,
            epochs=3,
            lr=2e-5,
            warmup_steps=0,
            gradient_accumulation_steps=1
    ):
        try:
            # load and preprocess datasets
            train_dataset = self.__load_local_dataset(train_set_path)
            val_dataset = self.__load_local_dataset(val_set_path)

            tokenized_train = train_dataset.map(
                self.__tokenize_and_align_labels,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            tokenized_val = val_dataset.map(
                self.__tokenize_and_align_labels,
                batched=True,
                remove_columns=val_dataset.column_names
            )

            collator = DataCollatorForTokenClassification(
                    tokenizer=self.tokenizer,
                    padding=True,
                    return_tensors='pt'
            )

            train_loader = DataLoader(
                tokenized_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=collator
            )
            val_loader = DataLoader(
                tokenized_val,
                batch_size=batch_size,
                collate_fn=collator
            )

            # initialize optimizer and scheduler
            optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=0.01  # add L2 regularization
            )

            num_training_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )

            best_val_loss = float('inf')
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

                optimizer.zero_grad()

                for step, batch in enumerate(train_pbar):
                    # move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps  # normalize loss
                    loss.backward()

                    if (step + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=1.0
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    total_loss += loss.item() * gradient_accumulation_steps
                    train_pbar.set_postfix({'loss': loss.item()})

                # validation phase
                val_loss, val_accuracy = self.evaluate(val_loader)
                avg_train_loss = total_loss / len(train_loader)

                # log metrics
                self.writer.add_scalars(
                    'Loss',
                    {'train': avg_train_loss, 'val': val_loss},
                    epoch
                )
                self.writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('best_model')

                logger.info(
                    f'Epoch {epoch+1}/{epochs} - '
                    f'Train Loss: {avg_train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, '
                    f'Val Accuracy: {val_accuracy:.4f}'
                )

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        eval_pbar = tqdm(data_loader, desc='Evaluating')

        for batch in eval_pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            total_loss += outputs.loss.item()

            # shape: batch_size x seq_length
            predictions = torch.argmax(outputs.logits, dim=2)
            # shape: batch_size x seq_length
            labels = batch['labels']

            # mask for valid tokens (ignore padding and special tokens)
            valid_mask = (labels != -100)

            # calculating accuracy only on valid tokens
            correct = ((predictions == labels) & valid_mask).sum().item()
            total = valid_mask.sum().item()

            total_correct += correct
            total_tokens += total

            # Update progress bar
            eval_pbar.set_postfix({
                'loss': outputs.loss.item(),
                'acc': correct/total if total > 0 else 0
            })

        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        return avg_loss, accuracy

    @torch.no_grad()
    def inference(self, texts):
        """
        Run inference on a list of texts to detect mountain names.

        Args:
            texts (list of str): A list of input sentences.

        Returns:
            list of dict: Each dictionary contains the tokenized text, corresponding labels, 
                          and word-label pairs.
        """
        self.model.eval()
        results = []

        for text in texts:
            words = text.split()
            tokenized_input = self.tokenizer(
                words,
                padding=True,
                truncation=True,
                return_tensors="pt",
                is_split_into_words=True
            ).to(self.device)

            # predict the label for each token and get word ids
            predictions = torch.argmax(self.model(**tokenized_input).logits, dim=2)[0].tolist()
            word_ids = tokenized_input.word_ids()

            # for word-label pairs
            word_label_list = []
            # for avoidance of duplication of pairs because of BPE
            seen_word_ids = set()
            for idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id not in seen_word_ids:
                    word_label_list.append((words[word_id], self.id2label[predictions[idx]]))
                    seen_word_ids.add(word_id)

            results.append({
                'text': words,
                'labels': [label for _, label in word_label_list],
                'word_label': word_label_list,
            })

        return results

    def save_model(self, dir_path):
        '''Save the model and tokenizer to the dir_path directory'''
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(dir_path)
            self.tokenizer.save_pretrained(dir_path)
            logger.info(f"Model saved to {dir_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, dir_path):
        '''Load the model and tokenizer from the dir_path directory'''
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.model = DistilBertForTokenClassification.from_pretrained(
                    dir_path,
                    ignore_mismatched_sizes=True
            )
            self.model.to(self.device)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                    dir_path,
                    ignore_mismatched_sizes=True
            )
            logger.info(f"Model saved to {dir_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def __load_local_dataset(self, file_path):
        """Load dataset from the file_path."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            texts, labels = [], []
            with open(file_path, 'r') as f:
                lines = f.read().strip().split('\n')

            for line in lines:
                try:
                    label_line, text_line = line.split('\t')
                    labels.append(label_line.split())
                    texts.append(text_line.split())
                except ValueError:
                    logger.warning(f"Skipping malformed line: {line}")
                    continue

            return Dataset.from_dict({"text": texts, "labels": labels})

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def __tokenize_and_align_labels(self, corpus):
        """Tokenize and align labels."""
        try:
            tokenized_inputs = self.tokenizer(
                    corpus["text"],
                    truncation=True,
                    is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(corpus["labels"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []

                for word_idx in word_ids:
                    if word_idx is None:
                        # set special tokens to -100
                        label_ids.append(-100)
                    else:
                        # label tokens
                        label_ids.append(self.label2id[label[word_idx]])
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            raise
