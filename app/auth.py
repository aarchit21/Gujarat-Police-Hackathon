import hmac

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


def _matches(supplied: str, expected: str) -> bool:
    """Constant-time comparison.

    `==` on a secret leaks its length and, in principle, its prefix through
    timing. The cost of doing this properly is one import.
    """
    if not supplied or not expected:
        return False
    return hmac.compare_digest(supplied, expected)


def require_operator(
    authorization: str | None = Header(default=None),
    x_operator_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> str:
    # `?token=` is accepted ONLY while the developer console is being served.
    #
    # The developer console needs it: `<img src>`, `<a href download>` and
    # `window.open` cannot carry an Authorization header, and console.js uses the
    # query form in nine places for crops, snapshots, CSV/GeoJSON exports and
    # report windows. The production console never does -- it fetches evidence
    # with the header and renders from a blob URL (app.js:291-312), precisely so
    # the token never reaches a URL.
    #
    # A credential in a query string lands in browser history, proxy logs and
    # Referer headers. That is an acceptable trade on a loopback-only developer
    # box and an unacceptable one on a public host, so the gate is the same flag
    # that decides whether /dev is served at all.
    if token and not settings.developer_ui_enabled():
        token = None
    supplied = _extract_token(authorization, x_operator_token, token)
    if _matches(supplied, settings.admin_token):
        return "operator"
    raise HTTPException(status_code=401, detail="unauthorised")


def require_vendor(authorization: str | None = Header(default=None)) -> str:
    supplied = _extract_token(authorization)
    if _matches(supplied, settings.vendor_ingest_token):
        return "vendor"
    raise HTTPException(status_code=401, detail="unauthorised vendor")
