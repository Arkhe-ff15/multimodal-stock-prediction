import re

file_path = "src/models.py"
backup_path = "src/models.py.bak"

# Read the file
with open(file_path, "r") as f:
    code = f.readlines()

# Backup the file
with open(backup_path, "w") as f:
    f.writelines(code)

# Patch all problematic lines
pattern = re.compile(r"^\s*predictions\s*=\s*torch\.tensor\(predictions,\s*dtype=torch\.float32\)")
fixed_code = []
for line in code:
    if pattern.match(line):
        indent = line[:len(line) - len(line.lstrip())]
        fixed_code.append(f"{indent}if not isinstance(predictions, torch.Tensor):\n")
        fixed_code.append(f"{indent}    predictions = torch.tensor(predictions, dtype=torch.float32)\n")
        fixed_code.append(f"{indent}else:\n")
        fixed_code.append(f"{indent}    predictions = predictions.float()\n")
    else:
        fixed_code.append(line)

# Write the fixed code back
with open(file_path, "w") as f:
    f.writelines(fixed_code)

print(f"✅ Patch applied to ALL occurrences! Backup saved as {backup_path}")
