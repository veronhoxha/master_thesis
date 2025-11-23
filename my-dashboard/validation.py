import os
from typing import Set


def validate_uploaded_file(uploaded_file, max_file_size_mb: int, allowed_extensions: Set[str]) -> str:
    """Validate extension and size of an uploaded file; return lowercase extension.

    Raises ValueError on invalid input.
    """
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {ext}")
    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_file_size_mb:
        raise ValueError(
            f"File too large: {size_mb:.1f} MB (max {max_file_size_mb} MB)"
        )
    return ext


