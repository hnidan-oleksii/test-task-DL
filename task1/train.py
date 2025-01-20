import argparse
import logging
from model import DistilBertMountainTokenClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training(
    train_set_path: str,
    val_set_path: str,
    model_path: str,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    max_length: int = 512
):
    """
    Train the DistilBERT model with specified parameters.

    Args:
        train_set_path: Path to training dataset
        val_set_path: Path to validation dataset
        model_path: Path to save the trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Warmup steps for scheduler
        gradient_accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length for tokenization
    """
    try:
        classifier = DistilBertMountainTokenClassifier(max_length=max_length)
        logger.info("Model initialized successfully")

        classifier.train(
            train_set_path=train_set_path,
            val_set_path=val_set_path,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        classifier.save_model(model_path)
        logger.info(f"Model training completed and saved to {model_path}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Train DistilBERT model on token classification task")
    parser.add_argument("--train_set_path", required=True, help="Path to the training dataset")
    parser.add_argument("--val_set_path", required=True, help="Path to the validation dataset")
    parser.add_argument("--model_path", required=True, help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps for the scheduler")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        train_set_path=args.train_set_path,
        val_set_path=args.val_set_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length
    )
