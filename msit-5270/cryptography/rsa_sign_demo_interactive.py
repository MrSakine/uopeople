import hashlib
import os
import sys

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# ----------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------

DIVIDER = "=" * 64
SUB = "-" * 64


def banner(title: str) -> None:
    """Print a section banner."""
    print()
    print(DIVIDER)
    print(f"  {title}")
    print(DIVIDER)


def section(title: str) -> None:
    """Print a sub-section divider."""
    print()
    print(SUB)
    print(f"  {title}")
    print(SUB)


def success(msg: str) -> None:
    print(f"  [OK]    {msg}")


def info(msg: str) -> None:
    print(f"  [INFO]  {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN]  {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL]  {msg}")


def prompt(question: str, default: str = "") -> str:
    """Prompt the user with an optional default value."""
    if default:
        answer = input(f"  > {question} [{default}]: ").strip()
        return answer if answer else default
    return input(f"  > {question}: ").strip()


def pause() -> None:
    input("\n  Press Enter to return to the menu...")


# ----------------------------------------------------------------------
# Cryptographic primitives
# ----------------------------------------------------------------------

def generate_keypair(key_size: int = 2048):
    """Generate an RSA keypair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    return private_key, private_key.public_key()


def save_keys(private_key, public_key, priv_path: str, pub_path: str) -> None:
    """Persist the keypair to PEM files."""
    with open(priv_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(pub_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ))


def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def hash_file(file_path: str) -> str:
    """Compute the SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def sign_file(file_path: str, private_key) -> bytes:
    """Sign a file's contents using RSA-PSS with SHA-256."""
    with open(file_path, "rb") as f:
        message = f.read()
    return private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )


def verify_signature(file_path: str, signature: bytes, public_key) -> bool:
    """Verify an RSA-PSS signature against a file's contents."""
    with open(file_path, "rb") as f:
        message = f.read()
    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False


# ----------------------------------------------------------------------
# Menu actions
# ----------------------------------------------------------------------

def action_generate_keypair() -> None:
    section("Generate RSA Keypair")
    priv_path = prompt("Private key output file", "private_key.pem")
    pub_path = prompt("Public key output file", "public_key.pem")
    info("Generating RSA-2048 keypair (this may take a moment)...")
    private_key, public_key = generate_keypair(2048)
    save_keys(private_key, public_key, priv_path, pub_path)
    success(f"Private key saved to: {priv_path}")
    success(f"Public  key saved to: {pub_path}")
    info(f"Key size: 2048 bits | Public exponent: 65537")
    pause()


def action_hash_file() -> None:
    section("Compute SHA-256 Hash")
    file_path = prompt("File to hash")
    if not os.path.isfile(file_path):
        fail(f"File not found: {file_path}")
        pause()
        return
    digest = hash_file(file_path)
    success(f"File: {file_path}")
    info(f"Size: {os.path.getsize(file_path)} bytes")
    info(f"SHA-256 digest:")
    print(f"          {digest}")
    pause()


def action_sign_file() -> None:
    section("Sign a File")
    file_path = prompt("File to sign")
    if not os.path.isfile(file_path):
        fail(f"File not found: {file_path}")
        pause()
        return
    priv_path = prompt("Private key file", "private_key.pem")
    if not os.path.isfile(priv_path):
        fail(f"Private key not found: {priv_path}")
        warn("Generate a keypair first using menu option 1.")
        pause()
        return
    sig_path = prompt("Output signature file", f"{file_path}.sig")

    private_key = load_private_key(priv_path)
    digest = hash_file(file_path)
    signature = sign_file(file_path, private_key)
    with open(sig_path, "wb") as f:
        f.write(signature)

    success(f"File signed: {file_path}")
    info(f"SHA-256 digest:")
    print(f"          {digest}")
    info(f"Signature length: {len(signature)} bytes")
    info(f"First 32 bytes (hex):")
    print(f"          {signature[:32].hex()}")
    success(f"Signature saved to: {sig_path}")
    pause()


def action_verify_signature() -> None:
    section("Verify a Signature")
    file_path = prompt("File to verify")
    if not os.path.isfile(file_path):
        fail(f"File not found: {file_path}")
        pause()
        return
    sig_path = prompt("Signature file", f"{file_path}.sig")
    if not os.path.isfile(sig_path):
        fail(f"Signature file not found: {sig_path}")
        pause()
        return
    pub_path = prompt("Public key file", "public_key.pem")
    if not os.path.isfile(pub_path):
        fail(f"Public key not found: {pub_path}")
        pause()
        return

    with open(sig_path, "rb") as f:
        signature = f.read()
    public_key = load_public_key(pub_path)
    is_valid = verify_signature(file_path, signature, public_key)

    if is_valid:
        success("Signature is VALID")
        info("The file has not been modified since it was signed,")
        info("and the signature was produced by the holder of the private key.")
    else:
        fail("Signature is INVALID")
        warn("Either the file was modified, the signature was tampered with,")
        warn("or the wrong public key is being used.")
    pause()


def action_tamper_test() -> None:
    section("Tamper Detection Test")
    info("This test will:")
    info("  1. Create a sample document")
    info("  2. Generate a keypair, sign the document")
    info("  3. Verify the signature on the ORIGINAL")
    info("  4. Modify the document by one line")
    info("  5. Verify the signature on the MODIFIED document")
    info("Expected result: original VALID, modified INVALID")
    print()

    sample_file = "tamper_test.txt"
    with open(sample_file, "w") as f:
        f.write("MRHN sample record\nPatient: Jane Doe\nMRN: 2026-001847\n")
    success(f"Created test document: {sample_file}")

    private_key, public_key = generate_keypair(2048)
    info("Generated RSA-2048 keypair (in-memory only)")

    signature = sign_file(sample_file, private_key)
    info(f"Signed document (signature length: {len(signature)} bytes)")

    if verify_signature(sample_file, signature, public_key):
        success("Verification on ORIGINAL document: VALID")
    else:
        fail("Unexpected: verification on original FAILED")

    with open(sample_file, "a") as f:
        f.write("Unauthorized addition by attacker.\n")
    warn("Document tampered with: line appended")

    if not verify_signature(sample_file, signature, public_key):
        success("Verification on TAMPERED document: INVALID  (tamper detected)")
    else:
        fail("Unexpected: tampered document still verified as VALID")

    os.remove(sample_file)
    info(f"Cleaned up test file: {sample_file}")
    pause()


def action_automated_demo() -> None:
    section("Automated Demo: Patient Consent Form")
    consent_text = (
        "MedSecure Regional Health Network\n"
        "Patient Consent for Treatment\n"
        "----------------------------------\n"
        "Patient Name: Jane Doe\n"
        "Medical Record Number: MRN-2026-001847\n"
        "Date: 2026-05-10\n\n"
        "I hereby authorize MedSecure Regional Health Network and its\n"
        "designated providers to administer the medical treatment\n"
        "described in this document. I acknowledge that I have been\n"
        "informed of the risks, benefits, and alternatives.\n\n"
        "Signed: Jane Doe\n"
    )
    sample_file = "patient_consent.txt"
    with open(sample_file, "w") as f:
        f.write(consent_text)
    success(f"Created sample document: {sample_file} ({os.path.getsize(sample_file)} bytes)")

    private_key, public_key = generate_keypair(2048)
    save_keys(private_key, public_key, "private_key.pem", "public_key.pem")
    success("Generated and saved RSA-2048 keypair")

    digest = hash_file(sample_file)
    info(f"SHA-256 digest:")
    print(f"          {digest}")

    signature = sign_file(sample_file, private_key)
    with open("consent_signature.bin", "wb") as f:
        f.write(signature)
    success(f"Signed document (256 bytes, saved to consent_signature.bin)")
    info(f"First 32 bytes of signature (hex):")
    print(f"          {signature[:32].hex()}")

    if verify_signature(sample_file, signature, public_key):
        success("Verification on ORIGINAL document: VALID  (integrity preserved)")

    with open(sample_file, "a") as f:
        f.write("\n[Unauthorized addition] Patient consents to additional procedures.\n")
    warn("Document tampered with: unauthorized line appended")

    if not verify_signature(sample_file, signature, public_key):
        success("Verification on TAMPERED document: INVALID  (tampering detected)")

    keep = prompt("Keep generated files for inspection? (y/n)", "n").lower()
    if keep != "y":
        for path in (sample_file, "consent_signature.bin",
                     "private_key.pem", "public_key.pem"):
            if os.path.exists(path):
                os.remove(path)
        info("Temporary files removed.")
    else:
        info("Files preserved in current directory.")
    pause()


# ----------------------------------------------------------------------
# Main menu
# ----------------------------------------------------------------------

def show_menu() -> str:
    print()
    print(DIVIDER)
    print("  MAIN MENU")
    print(DIVIDER)
    print("  1) Generate RSA keypair")
    print("  2) Compute SHA-256 hash of a file")
    print("  3) Sign a file with RSA-PSS")
    print("  4) Verify a signature")
    print("  5) Run automated demo (patient consent form)")
    print("  6) Run tamper detection test")
    print("  7) Exit")
    print()
    return prompt("Enter choice [1-7]")


def main() -> None:
    banner("RSA Hashing and Digital Signature Tool")
    print("  MedSecure Regional Health Network (MRHN)")
    print("  Week 5 Cryptography Demonstration")
    print()
    info("Algorithms: RSA-2048 keypair, SHA-256 hashing, RSA-PSS signatures")
    info("Library:    cryptography (Python)")

    actions = {
        "1": action_generate_keypair,
        "2": action_hash_file,
        "3": action_sign_file,
        "4": action_verify_signature,
        "5": action_automated_demo,
        "6": action_tamper_test,
    }

    while True:
        choice = show_menu()
        if choice == "7" or choice.lower() in ("exit", "quit", "q"):
            print()
            success("Goodbye.")
            print()
            sys.exit(0)
        action = actions.get(choice)
        if action:
            action()
        else:
            warn(f"Invalid choice: {choice!r}. Please enter 1-7.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        warn("Interrupted by user. Exiting.")
        sys.exit(0)
