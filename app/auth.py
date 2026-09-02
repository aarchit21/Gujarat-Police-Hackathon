from fastapi import Header, HTTPException, Query

from app.config import settings


def _extract_token(authorization: str | None, x_operator_token: str | None = None, token: str | None = None) -> str:
    if x_operator_token:
        return x_operator_token
    if token:
        return token
    if authorization:
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return authorization.strip()
    return ""


def require_operator(
    authorization: str | None = Header(default=None),
    x_operator_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> str:
    supplied = _extract_token(authorization, x_operator_token, token)
    if supplied == settings.admin_token:
        return "operator"
    if not settings.require_auth and not supplied:
        return "operator"
    raise HTTPException(status_code=401, detail="unauthorised")


def require_vendor(authorization: str | None = Header(default=None)) -> str:
    supplied = _extract_token(authorization)
    if supplied == settings.vendor_ingest_token:
        return "vendor"
    raise HTTPException(status_code=401, detail="unauthorised vendor")
