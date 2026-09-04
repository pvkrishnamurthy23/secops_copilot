# services/auth_service.py

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class User:
    id: str
    email: str
    password_hash: str
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: Optional[float] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class AuthToken:
    token: str
    user_id: str
    expires_at: float
    scope: str = "read"


class AuthenticationError(Exception):
    pass


class AccountLockedException(Exception):
    pass


class AuthService:
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION = 300  # 5 minutes in seconds
    TOKEN_TTL = 3600  # 1 hour

    def __init__(self, secret_key: str):
        self._secret = secret_key.encode()
        self._users: dict[str, User] = {}
        self._tokens: dict[str, AuthToken] = {}

    def register(self, email: str, password: str) -> User:
        """Register a new user with hashed password."""
        if self._find_user_by_email(email):
            raise ValueError(f"User with email {email} already exists.")

        user_id = secrets.token_hex(16)
        password_hash = self._hash_password(password)

        user = User(id=user_id, email=email, password_hash=password_hash)
        self._users[user_id] = user
        return user

    def login(self, email: str, password: str) -> AuthToken:
        """Authenticate a user and return a session token."""
        user = self._find_user_by_email(email)
        if not user:
            raise AuthenticationError("Invalid email or password.")

        if not user.is_active:
            raise AuthenticationError("Account is disabled.")

        if self._is_locked(user):
            remaining = int(user.locked_until - time.time())
            raise AccountLockedException(
                f"Account locked. Try again in {remaining} seconds."
            )

        if not self._verify_password(password, user.password_hash):
            user.failed_attempts += 1
            if user.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked_until = time.time() + self.LOCKOUT_DURATION
                raise AccountLockedException("Account locked due to too many failed attempts.")
            raise AuthenticationError("Invalid email or password.")

        # Successful login — reset counters
        user.failed_attempts = 0
        user.locked_until = None

        return self._issue_token(user)

    def validate_token(self, token: str) -> Optional[User]:
        """Validate a token and return the associated user."""
        auth_token = self._tokens.get(token)
        if not auth_token:
            return None
        if time.time() > auth_token.expires_at:
            del self._tokens[token]
            return None
        return self._users.get(auth_token.user_id)

    def logout(self, token: str) -> bool:
        """Revoke a session token."""
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False

    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change a user's password after verifying old password."""
        user = self._users.get(user_id)
        if not user:
            raise AuthenticationError("User not found.")
        if not self._verify_password(old_password, user.password_hash):
            raise AuthenticationError("Current password is incorrect.")
        user.password_hash = self._hash_password(new_password)
        return True

    # --- Private helpers ---

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        hashed = hmac.new(self._secret, (salt + password).encode(), hashlib.sha256).hexdigest()
        return f"{salt}:{hashed}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        salt, hashed = stored_hash.split(":", 1)
        expected = hmac.new(self._secret, (salt + password).encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, hashed)

    def _issue_token(self, user: User) -> AuthToken:
        token_str = secrets.token_urlsafe(32)
        auth_token = AuthToken(
            token=token_str,
            user_id=user.id,
            expires_at=time.time() + self.TOKEN_TTL
        )
        self._tokens[token_str] = auth_token
        return auth_token

    def _find_user_by_email(self, email: str) -> Optional[User]:
        return next((u for u in self._users.values() if u.email == email), None)

    def _is_locked(self, user: User) -> bool:
        if user.locked_until and time.time() < user.locked_until:
            return True
        return False
