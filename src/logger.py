import logging
import os
from datetime import datetime

# Log file name
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Logs directory
log_dir = os.path.join(os.getcwd(), "logs", log_file)

# Create logs directory if not exists
os.makedirs(log_dir, exist_ok=True)

# Full log file path
log_file_path = os.path.join(log_dir, log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started.")
