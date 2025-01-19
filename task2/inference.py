import argparse
import logging
from matcher import LightGlue_DISK_Matcher
from torch import device as torch_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_inference(image1_path, image2_path, output_path=None, image_size=None):
    """
    Run image matching inference using LightGlue and DISK.

    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        output_path: Path to save the output visualization
    """
    try:
        device = torch_device('cpu')
        if image_size and not isinstance(image_size, tuple):
            image_size = tuple(image_size)

        logger.info("Starting inference")
        matcher = LightGlue_DISK_Matcher(device=device, image_size=image_size)
        result = matcher.match(image1_path, image2_path)
        matcher.draw_matches(**result, output=output_path)
        logger.info(f"Inference completed. Output visualization saved to {output_path}")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run image matching inference")
    parser.add_argument("--image1_path", required=True, help="Path to the first image")
    parser.add_argument("--image2_path", required=True, help="Path to the second image")
    parser.add_argument("--output_path", help="Path to save the output visualization")
    parser.add_argument("--image_size", type=int, nargs=2, metavar=("height", "width"), help="Image size after scaling in form height x width")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        image1_path=args.image1_path,
        image2_path=args.image2_path,
        output_path=args.output_path,
        image_size=args.image_size
    )
