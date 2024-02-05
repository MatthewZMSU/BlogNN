from telethon import TelegramClient
from security import decrypt_file

import configparser
import json

CHANNEL_TITLE = "Ботаем вместе"
MESSAGES_FILE = "data/messages.json"
CONFIG_FILE = "data/telegram_config.ini"


def main():
    config = configparser.ConfigParser()
    config.read_string(decrypt_file(CONFIG_FILE))

    client = TelegramClient(config['Telegram']['username'],
                            int(config['Telegram']['api_id']),
                            config['Telegram']['api_hash'])
    client.start()

    for dialogue in client.iter_dialogs():
        if dialogue.title == CHANNEL_TITLE:
            break
    else:
        raise KeyError(f"No such channel: {CHANNEL_TITLE}")

    all_messages = []
    for message in client.iter_messages(dialogue):
        all_messages.append({
            'author': message.post_author,
            'message': message.message,
        })

    with open(MESSAGES_FILE, 'w') as f:
        json.dump(all_messages, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
