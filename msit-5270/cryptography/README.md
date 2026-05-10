# RSA Hashing and Digital Signature Tool

An interactive CLI for demonstrating SHA-256 hashing and RSA-PSS digital signatures. Users can generate key pairs, sign and verify any file, and run automated demonstrations of integrity and tamper detection.

## Dependencies

```bash
pip install cryptography
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## How to Run

```bash
python rsa_sign_demo_interactive.py
```

---

## Menu Options Guide

### Option 1: Generate RSA Keypair

**Purpose:** Generate a 2048-bit RSA keypair and save it to PEM files.

**Inputs Required:**
- Private key output file (default: `private_key.pem`)
- Public key output file (default: `public_key.pem`)

**Process:**
- Generates RSA-2048 keypair with public exponent 65537
- Saves private key in PKCS8 format (PEM encoding)
- Saves public key in SubjectPublicKeyInfo format (PEM encoding)
- Keys are saved without encryption

**Expected Output:**
```
Private key saved to: private_key.pem
Public key saved to: public_key.pem
Key size: 2048 bits | Public exponent: 65537
```

**How to Test:**
1. Run the script and select option `1`
2. Press Enter to accept default filenames or provide custom names
3. Verify that two `.pem` files are created in the directory
4. Check that `private_key.pem` contains `-----BEGIN PRIVATE KEY-----`
5. Check that `public_key.pem` contains `-----BEGIN PUBLIC KEY-----`

**Example Interaction:**
```
> Private key output file [private_key.pem]: 
> Public key output file [public_key.pem]: 
[OK]    Private key saved to: private_key.pem
[OK]    Public key saved to: public_key.pem
[INFO]  Key size: 2048 bits | Public exponent: 65537
```

---

### Option 2: Compute SHA-256 Hash of a File

**Purpose:** Compute and display the SHA-256 hash digest of any file.

**Inputs Required:**
- File path to hash (any file)

**Process:**
- Reads file in 8192-byte chunks
- Computes SHA-256 hash digest
- Displays file size and hexadecimal digest

**Expected Output:**
```
File: patient_consent.txt
Size: 493 bytes
SHA-256 digest:
          a3f8b92c... (64-character hex string)
```

**How to Test:**
1. Create a test file or use existing file (e.g., `patient_consent.txt`)
2. Run the script and select option `2`
3. Enter the file path when prompted
4. Note the SHA-256 hash (64 hexadecimal characters)
5. Modify the file and hash again - the digest should change
6. Restore the file to original state - digest should match original

**Example Interaction:**
```
> File to hash: patient_consent.txt
[OK]    File: patient_consent.txt
[INFO]  Size: 493 bytes
[INFO]  SHA-256 digest:
          a3f8b92c7e4d1a5f2b8c9e3d7a1f4b6e8c2d5a9f3e7b1c4d8a2f6e9c3b7d1a5f
```

**Test Files to Use:**
- `patient_consent.txt` (if created by option 5)
- Any text file in the directory
- The Python script itself

---

### Option 3: Sign a File with RSA-PSS

**Purpose:** Create a digital signature for a file using RSA-PSS with SHA-256.

**Inputs Required:**
- File to sign
- Private key file (default: `private_key.pem`)
- Output signature file (default: `[filename].sig`)

**Process:**
- Reads the file contents
- Computes SHA-256 hash
- Signs using RSA-PSS with MGF1(SHA-256) and maximum salt length
- Saves signature to binary file

**Expected Output:**
```
File signed: patient_consent.txt
SHA-256 digest:
          a3f8b92c... (64 characters)
Signature length: 256 bytes
First 32 bytes (hex):
          8a3f2b1c...
Signature saved to: patient_consent.txt.sig
```

**How to Test:**
1. First generate a keypair using option `1`
2. Create or use an existing file to sign
3. Run the script and select option `3`
4. Enter the file path, private key path, and signature output path
5. Verify that a `.sig` file is created
6. Check that signature file is 256 bytes (for RSA-2048)

**Example Interaction:**
```
> File to sign: patient_consent.txt
> Private key file [private_key.pem]: 
> Output signature file [patient_consent.txt.sig]: 
[OK]    File signed: patient_consent.txt
[INFO]  SHA-256 digest:
          a3f8b92c7e4d1a5f2b8c9e3d7a1f4b6e8c2d5a9f3e7b1c4d8a2f6e9c3b7d1a5f
[INFO]  Signature length: 256 bytes
[INFO]  First 32 bytes (hex):
          8a3f2b1c4e5d6a7f9c8e3b2d1a5f4e6c...
[OK]    Signature saved to: patient_consent.txt.sig
```

**Prerequisites:**
- Must have a private key (generate using option 1)
- File to sign must exist

---

### Option 4: Verify a Signature

**Purpose:** Verify the authenticity and integrity of a signed file using its signature and public key.

**Inputs Required:**
- File to verify
- Signature file (default: `[filename].sig`)
- Public key file (default: `public_key.pem`)

**Process:**
- Reads file contents and signature
- Loads public key
- Verifies signature using RSA-PSS with SHA-256
- Returns VALID or INVALID result

**Expected Output (Valid):**
```
Signature is VALID
The file has not been modified since it was signed,
and the signature was produced by the holder of the private key.
```

**Expected Output (Invalid):**
```
Signature is INVALID
Either the file was modified, the signature was tampered with,
or the wrong public key is being used.
```

**How to Test - Valid Signature:**
1. Sign a file using option `3`
2. Run the script and select option `4`
3. Enter the file path, signature path, and public key path
4. Should show "Signature is VALID"

**How to Test - Invalid Signature (Tampered File):**
1. Sign a file using option `3`
2. Modify the original file (add/remove text)
3. Run option `4` to verify
4. Should show "Signature is INVALID"

**How to Test - Invalid Signature (Wrong Key):**
1. Sign a file using option `3` with keypair A
2. Generate a different keypair (keypair B)
3. Try to verify using public key from keypair B
4. Should show "Signature is INVALID"

**Example Interaction (Valid):**
```
> File to verify: patient_consent.txt
> Signature file [patient_consent.txt.sig]: 
> Public key file [public_key.pem]: 
[OK]    Signature is VALID
[INFO]  The file has not been modified since it was signed,
[INFO]  and the signature was produced by the holder of the private key.
```

**Example Interaction (Invalid - Tampered):**
```
> File to verify: patient_consent.txt
> Signature file [patient_consent.txt.sig]: 
> Public key file [public_key.pem]: 
[FAIL]  Signature is INVALID
[WARN]  Either the file was modified, the signature was tampered with,
[WARN]  or the wrong public key is being used.
```

---

### Option 5: Run Automated Demo (Patient Consent Form)

**Purpose:** Demonstrate complete workflow of signing and tamper detection using a patient consent form scenario.

**Inputs Required:**
- Keep files option: `y` or `n` (default: `n`)

**Process:**
1. Creates sample `patient_consent.txt` with medical consent content
2. Generates new RSA-2048 keypair
3. Saves keypair to `private_key.pem` and `public_key.pem`
4. Computes and displays SHA-256 hash
5. Signs document and saves to `consent_signature.bin`
6. Verifies signature on original (shows VALID)
7. Tampers with document by appending unauthorized text
8. Verifies signature on tampered document (shows INVALID)
9. Optionally cleans up or keeps generated files

**Expected Output:**
```
Created sample document: patient_consent.txt (493 bytes)
Generated and saved RSA-2048 keypair
SHA-256 digest:
          a3f8b92c...
Signed document (256 bytes, saved to consent_signature.bin)
First 32 bytes of signature (hex):
          8a3f2b1c...
Verification on ORIGINAL document: VALID (integrity preserved)
Document tampered with: unauthorized line appended
Verification on TAMPERED document: INVALID (tampering detected)
```

**How to Test:**
1. Run the script and select option `5`
2. Observe the complete workflow execution
3. When prompted, choose `y` to keep files for inspection
4. Examine generated files:
   - `patient_consent.txt` (tampered version)
   - `consent_signature.bin` (binary signature)
   - `private_key.pem` and `public_key.pem`
5. Try manually verifying with option `4` - should show INVALID due to tampering

**Files Created:**
- `patient_consent.txt` - Medical consent form (tampered at end)
- `consent_signature.bin` - 256-byte signature file
- `private_key.pem` - RSA private key
- `public_key.pem` - RSA public key

**Example Interaction:**
```
[OK]    Created sample document: patient_consent.txt (493 bytes)
[OK]    Generated and saved RSA-2048 keypair
[INFO]  SHA-256 digest:
          a3f8b92c7e4d1a5f2b8c9e3d7a1f4b6e8c2d5a9f3e7b1c4d8a2f6e9c3b7d1a5f
[OK]    Signed document (256 bytes, saved to consent_signature.bin)
[INFO]  First 32 bytes of signature (hex):
          8a3f2b1c4e5d6a7f9c8e3b2d1a5f4e6c...
[OK]    Verification on ORIGINAL document: VALID (integrity preserved)
[WARN]  Document tampered with: unauthorized line appended
[OK]    Verification on TAMPERED document: INVALID (tampering detected)
> Keep generated files for inspection? (y/n) [n]: y
[INFO]  Files preserved in current directory.
```

---

### Option 6: Run Tamper Detection Test

**Purpose:** Automated test demonstrating tamper detection capabilities without keeping files.

**Inputs Required:**
- None (fully automated)

**Process:**
1. Creates test document `tamper_test.txt` with sample MRHN record
2. Generates RSA-2048 keypair (in-memory only, not saved)
3. Signs the document
4. Verifies signature on original document (VALID)
5. Appends line to document to simulate tampering
6. Verifies signature on tampered document (INVALID)
7. Cleans up test file automatically

**Expected Output:**
```
This test will:
  1. Create a sample document
  2. Generate a keypair, sign the document
  3. Verify the signature on the ORIGINAL
  4. Modify the document by one line
  5. Verify the signature on the MODIFIED document
Expected result: original VALID, modified INVALID

Created test document: tamper_test.txt
Generated RSA-2048 keypair (in-memory only)
Signed document (signature length: 256 bytes)
Verification on ORIGINAL document: VALID
Document tampered with: line appended
Verification on TAMPERED document: INVALID (tamper detected)
Cleaned up test file: tamper_test.txt
```

**How to Test:**
1. Run the script and select option `6`
2. Observe the automated test execution
3. Verify that both checks produce expected results:
   - Original: VALID ✓
   - Tampered: INVALID ✓
4. Confirm no files remain after test completes

**Example Interaction:**
```
[INFO]  This test will:
[INFO]    1. Create a sample document
[INFO]    2. Generate a keypair, sign the document
[INFO]    3. Verify the signature on the ORIGINAL
[INFO]    4. Modify the document by one line
[INFO]    5. Verify the signature on the MODIFIED document
[INFO]  Expected result: original VALID, modified INVALID

[OK]    Created test document: tamper_test.txt
[INFO]  Generated RSA-2048 keypair (in-memory only)
[INFO]  Signed document (signature length: 256 bytes)
[OK]    Verification on ORIGINAL document: VALID
[WARN]  Document tampered with: line appended
[OK]    Verification on TAMPERED document: INVALID (tamper detected)
[INFO]  Cleaned up test file: tamper_test.txt
```

**Use Case:** Quick verification that the RSA-PSS signature system correctly detects document tampering.

---

### Option 7: Exit

**Purpose:** Exit the application.

**How to Use:**
- Select option `7`, or
- Type `exit`, `quit`, or `q`
- Press `Ctrl+C` to interrupt

---

## Complete Testing Workflow

### Basic Workflow Test (Manual Steps)

This test demonstrates the full signing and verification process manually:

```bash
# Step 1: Generate keypair
Select option: 1
Private key: private_key.pem (default)
Public key: public_key.pem (default)

# Step 2: Create a test file
echo "Important medical record for Patient XYZ" > test_document.txt

# Step 3: Hash the file (optional, for reference)
Select option: 2
File to hash: test_document.txt
# Note the hash value

# Step 4: Sign the file
Select option: 3
File to sign: test_document.txt
Private key: private_key.pem (default)
Signature file: test_document.txt.sig (default)

# Step 5: Verify signature (should be VALID)
Select option: 4
File to verify: test_document.txt
Signature file: test_document.txt.sig (default)
Public key: public_key.pem (default)
# Result: VALID ✓

# Step 6: Tamper with the file
echo "Unauthorized modification" >> test_document.txt

# Step 7: Verify signature again (should be INVALID)
Select option: 4
File to verify: test_document.txt
Signature file: test_document.txt.sig (default)
Public key: public_key.pem (default)
# Result: INVALID ✓
```

### Quick Automated Test

For a quick demonstration without manual steps:

```bash
# Run automated demo
Select option: 5
Keep files: n (will clean up automatically)

# Or run tamper detection test
Select option: 6
(Fully automated, no input required)
```

### Hash Verification Test

Test that identical files produce identical hashes:

```bash
# Create two identical files
echo "Test content" > file1.txt
cp file1.txt file2.txt

# Hash both files (option 2)
# Both should produce identical SHA-256 hashes

# Modify one file
echo "Extra line" >> file2.txt

# Hash again - file2 should have different hash
```

### Cross-Verification Test

Test that signatures are specific to their documents:

```bash
# Generate keypair (option 1)
# Create two different files
echo "Document A" > docA.txt
echo "Document B" > docB.txt

# Sign document A (option 3)
# Input: docA.txt, signature: docA.txt.sig

# Sign document B (option 3)
# Input: docB.txt, signature: docB.txt.sig

# Verify docA with docA.sig (option 4) - VALID ✓
# Try to verify docA with docB.sig (option 4) - INVALID ✓
# Try to verify docB with docA.sig (option 4) - INVALID ✓
```

---

## Algorithm Details

### Cryptographic Algorithms Used

- **RSA Key Generation:**
  - Key size: 2048 bits
  - Public exponent: 65537
  - Algorithm: RSA

- **Hashing:**
  - Algorithm: SHA-256
  - Output: 256-bit (32-byte) digest
  - Encoding: Hexadecimal string (64 characters)

- **Digital Signature:**
  - Scheme: RSA-PSS (Probabilistic Signature Scheme)
  - Hash function: SHA-256
  - Mask generation function: MGF1 with SHA-256
  - Salt length: Maximum (depends on key size)
  - Signature size: 256 bytes (for 2048-bit RSA)

### Security Properties

- **Integrity:** Any modification to signed document will invalidate signature
- **Authenticity:** Only holder of private key can create valid signature
- **Non-repudiation:** Signer cannot deny creating the signature
- **Tamper Detection:** Even single-byte changes are detected

---

## File Format Reference

### Generated Files

| File | Format | Size | Description |
|------|--------|------|-------------|
| `private_key.pem` | PEM (PKCS8) | ~1.7 KB | RSA private key, unencrypted |
| `public_key.pem` | PEM (SubjectPublicKeyInfo) | ~450 bytes | RSA public key |
| `*.sig` | Binary | 256 bytes | RSA-PSS signature (for 2048-bit key) |
| `patient_consent.txt` | Text | ~493 bytes | Sample medical consent form |
| `consent_signature.bin` | Binary | 256 bytes | Signature for consent form |

### PEM File Structure

**Private Key (`private_key.pem`):**
```
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...
(base64 encoded key data)
...
-----END PRIVATE KEY-----
```

**Public Key (`public_key.pem`):**
```
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...
(base64 encoded key data)
...
-----END PUBLIC KEY-----
```

---

## Troubleshooting

### Common Issues

**Issue:** "File not found" error
- **Solution:** Ensure file path is correct relative to current directory
- Check that file actually exists using `ls` or file explorer

**Issue:** "Private key not found" when signing
- **Solution:** Generate keypair first using option 1
- Verify `private_key.pem` exists in directory

**Issue:** Signature verification always shows INVALID
- **Solution:** 
  - Ensure you're using the matching public key for the private key that signed
  - Verify the file hasn't been modified since signing
  - Check that signature file is correct and not corrupted

**Issue:** Permission denied when creating files
- **Solution:** Ensure you have write permissions in current directory
- Try running from a different directory

---

## Use Cases

### Medical Records Security
- Sign patient consent forms to ensure integrity
- Detect unauthorized modifications to medical records
- Maintain audit trail of document authenticity

### Document Authentication
- Sign important documents before distribution
- Verify document hasn't been altered
- Prove document origin and authenticity

### Educational Demonstrations
- Understand RSA-PSS signature scheme
- Learn about hash functions and digital signatures
- Demonstrate tamper detection capabilities

---

## Additional Notes

- **Key Storage:** Private keys are saved without password protection. In production, use password-encrypted keys.
- **Key Reuse:** You can reuse generated keypairs for multiple documents
- **File Size:** Can sign files of any size (memory permitting)
- **Hash Uniqueness:** Each unique file content produces unique SHA-256 hash
- **Signature Uniqueness:** RSA-PSS includes randomness, so same file produces different signatures each time (but all verify correctly)

---

## Quick Reference

| Task | Menu Options Needed |
|------|-------------------|
| First-time setup | Option 1 (generate keypair) |
| Sign a document | Options 1 → 3 |
| Verify a document | Option 4 |
| Quick demo | Option 5 or 6 |
| Check file hash | Option 2 |
| Test tamper detection | Options 3 → 4 → modify file → 4 |

---

## Project Information

**Course:** MSIT-5270 Cryptography  
**Week:** 5  
**Topic:** RSA Digital Signatures and Hash Functions  
**Organization:** MedSecure Regional Health Network (MRHN)  
**Library:** Python `cryptography` package
