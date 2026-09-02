"""
models.py — Anchor AI V2 authentication data model.

Holds the SQLAlchemy `db` instance and the `User` model. `db` lives here
(rather than in app.py) so that both app.py and auth.py can import it
without creating a circular import between the two.
"""

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """
    A registered Anchor AI user.

    Password is never stored in plaintext — only `password_hash`
    (bcrypt, written/verified via Flask-Bcrypt in auth.py) is persisted.
    """

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return f"<User id={self.id} email={self.email!r}>"
