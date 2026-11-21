class LakehouseAuthError(RuntimeError):
    """AWS credentials are missing, expired, or point at the wrong account."""
