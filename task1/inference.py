import argparse
import json
import logging
from tqdm import tqdm

from model import DistilBertMountainTokenClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_inference(
    model_path,
    input_path,
    output_path,
    batch_size=256
):
    """
    Run NER inference on texts from input file and save predictions.

    Args:
        model_path: Path to the saved model
        input_path: Path to input file (one text per line)
        output_path: Path to save predictions
        batch_size: Batch size for inference
    """
    try:
        classifier = DistilBertMountainTokenClassifier()
        classifier.load_model(model_path)
        logger.info("Model loaded successfully")

        with open(input_path) as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} texts")

        # Run inference in batches
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            result = classifier.inference(batch_texts)
            results.extend(result)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on text data")
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--input_path", required=True, help="Path to input texts")
    parser.add_argument("--output_path", required=True, help="Path to save predictions")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=args.batch_size
    )
