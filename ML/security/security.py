from cryptography.fernet import Fernet
from pathlib import Path
from typing import Generator

SECURITY_FOLDER = Path("./security")
KEY_FILE = SECURITY_FOLDER / "key.key"
FILES_TO_ENCRYPT = SECURITY_FOLDER / "files_to_encrypt"


def __generate_key():
    if KEY_FILE.exists():
        raise FileExistsError("Key already exists")
    with open(KEY_FILE, "wb") as f:
        secret_key = Fernet.generate_key()
        f.write(secret_key)


def __load_key() -> bytes:
    try:
        return open(KEY_FILE, "rb").read()
    except FileNotFoundError:
        print("Key file not found in security folder.")
        raise
    except Exception:
        raise


key = __load_key()
encoder = Fernet(key)


def __encrypt_file(filename: str | Path):
    try:
        with open(filename, "rb") as f:
            file_data = f.read()
        encrypted_data = encoder.encrypt(file_data)
        with open(filename, "wb") as f:
            f.write(encrypted_data)
    except FileNotFoundError:
        print(f"Check file to be encrypted: {filename}.")
        raise
    except Exception:
        raise


def decrypt_file(filename: str | Path) -> str:
    with open(filename, "rb") as f:
        file_data = f.read()
    decrypted_data = encoder.decrypt(file_data)
    return decrypted_data.decode()


def __get_encryption_files() -> Generator[Path, None, None]:
    with open(FILES_TO_ENCRYPT, "r") as f:
        for line in f:
            yield Path(line.strip())


def main():
    print("Starting security process")
    try:
        __generate_key()
        print("Generating/getting security key")
    except FileExistsError:
        pass
    except Exception:
        raise

    print("Encrypting files")
    try:
        encryption_files = __get_encryption_files()
    except FileNotFoundError:
        print("Check files_to_encrypt.")
        raise
    except Exception:
        raise

    for file_path in encryption_files:
        if not file_path.exists():
            raise FileNotFoundError(f"No file to encrypt: {file_path}")
        __encrypt_file(file_path)

    print("Encryption successful!")
