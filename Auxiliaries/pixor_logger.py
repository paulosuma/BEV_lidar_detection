import logging    

logger = logging.getLogger("pixor_logger")
logger.setLevel(logging.DEBUG)

# StreamHandler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# FileHandler
file_handler = logging.FileHandler('Pixor_logger/kitti_logger.log')
file_handler.setLevel(logging.ERROR)

# Create formatters and add them to handlers
console_formatter = logging.Formatter('%(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


logger.info("paul".format(34))