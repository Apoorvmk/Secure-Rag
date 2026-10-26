"""Create a company login without exposing the password in shell history."""

import os
from getpass import getpass
from pathlib import Path

import bcrypt
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from access_policy import ROLES
from models import Base, User


def prompt_required(label: str) -> str:
    while True:
        value = input(label).strip()
        if value:
            return value
        print("This field is required.")


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is missing from Backend/.env")

    print("Create a SecureRAG company account")
    employee_id = prompt_required("Employee ID: ")
    name = prompt_required("Full name: ")
    email = prompt_required("Company email: ").lower()
    print("Role options: " + ", ".join(sorted(ROLES)))
    role = prompt_required("Role: ").lower()
    if role not in ROLES:
        raise SystemExit("Unknown role. Choose intern, employee, manager, or board.")

    password = getpass("Password (8–72 UTF-8 bytes): ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")
    if len(password) < 8 or len(password.encode("utf-8")) > 72:
        raise SystemExit("Password must be at least 8 characters and no more than 72 UTF-8 bytes.")

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        Base.metadata.create_all(bind=engine)
        with Session(engine) as session:
            if session.query(User).filter(
                (User.email == email) | (User.employee_id == employee_id)
            ).first():
                raise SystemExit("An account already uses that email or employee ID.")
            user = User(
                employee_id=employee_id,
                name=name,
                email=email,
                password_hash=bcrypt.hashpw(
                    password.encode("utf-8"), bcrypt.gensalt()
                ).decode("utf-8"),
                role=role,
                is_active=True,
            )
            session.add(user)
            session.commit()
        print(f"Created active {role} account for {email}.")
    except IntegrityError as exc:
        raise SystemExit("An account already uses that email or employee ID.") from exc
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
