from security import decrypt_file

import configparser
import json


print(decrypt_file("data/telegram_config.ini"))
# cnf_parser = configparser.ConfigParser()
# cnf_parser.read_string(decrypt_file("data/telegram_config.ini"))
