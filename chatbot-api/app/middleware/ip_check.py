# app/middleware/ip_check.py
import ipaddress
from typing import List, Union

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_403_FORBIDDEN

from app.core.config import ALLOWED_IPS, CHATBOT_INVOKE_API_KEY

# description: _parse_allowlist parses the ALLOWED_IPS list into a list of ipaddress objects (supports both IPs and CIDR)
# takes params:
    # raw: the ALLOWED_IPS list (list of strings) (example: ["127.0.0.1", "192.168.1.0/24"])
# returns:
    # a list of ipaddress objects
def _parse_allowlist(raw: List[str]) -> List[Union[ipaddress.IPv4Address, ipaddress.IPv4Network,
                                                    ipaddress.IPv6Address, ipaddress.IPv6Network]]:
    """Parse ALLOWED_IPS into address/network objects (supports both IPs and CIDR)."""
    parsed = []
    for entry in raw:
        entry = entry.strip()
        if not entry:
            continue
        try:
            if "/" in entry:
                parsed.append(ipaddress.ip_network(entry, strict=False))
            else:
                parsed.append(ipaddress.ip_address(entry))
        except ValueError:
            pass
    return parsed


_PARSED_ALLOWLIST = _parse_allowlist(ALLOWED_IPS) if ALLOWED_IPS else []

# description: _is_allowed checks if the client IP is in the allowed list (helper function for dispatch method)
# takes params:
    # ip_str: the client IP address 
# returns:
    # True if the client IP is in the allowed list
    # False if the client IP is not in the allowed list
def _is_allowed(ip_str: str) -> bool:
    if not _PARSED_ALLOWLIST:
        return True
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    for entry in _PARSED_ALLOWLIST:
        if isinstance(entry, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
            if addr in entry:
                return True
        elif addr == entry:
            return True
    return False

# description: IPAllowlistMiddleware is a middleware that checks if the client IP is in the allowed list
# class IPAllowlistMiddleware is a subclass of BaseHTTPMiddleware
# it uses:
# dispatch method to check if the client IP is in the allowed list
class IPAllowlistMiddleware(BaseHTTPMiddleware):
    # The chatbot gateway is meant to be called from anywhere, but it only
    # leaves the allowlist once an API key exists to authenticate those callers.
    PUBLIC_PREFIXES = ("/api/deep-link/", "/api/doc/") + (
        ("/api/chatbot/",) if CHATBOT_INVOKE_API_KEY else ()
    )
    # description: dispatch method checks if the client IP is in the allowed list
        # if it is not in the allowed list, it returns a Response object with a status code of 403
        # if it is in the allowed list, it calls the next middleware or endpoint (async function that returns a Response object)
    # takes params:
        # request: the request object
        # call_next: call the next middleware or return the response from pipeline
    # returns:
        # Response object if the client IP is not in the allowed list
        # await call_next(request) if the client IP is in the allowed list
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        upgrade = request.headers.get("upgrade", "").lower()
        if upgrade == "websocket":
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(prefix) for prefix in self.PUBLIC_PREFIXES):
            return await call_next(request)

        client_ip = request.client.host
        real_ip = request.headers.get("x-real-ip", client_ip)
        # check if the client IP is in the allowed list
        if _PARSED_ALLOWLIST:
            if not _is_allowed(client_ip) and not _is_allowed(real_ip):
                return Response(f"Access Denied for IP address: {client_ip}", status_code=HTTP_403_FORBIDDEN)
        return await call_next(request)