from src.service.cli import app
import sys

if __name__ == "__main__":
    sys.argv = ["car_price", "train"] + sys.argv[1:]
    app()