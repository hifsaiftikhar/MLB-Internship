import sys
from pathlib import Path

# Add project root to sys.path so tests can be run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database.database import Base, get_db
from app.database import crud
from app.database.models import User, Job

# Set up an isolated in-memory SQLite database for test suite
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Create database tables before running tests and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


# =======================================================
# 1. Health & Root Endpoint Tests
# =======================================================

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "authentication" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


# =======================================================
# 2. User Registration Tests
# =======================================================

def test_register_new_user_success():
    """Case: Register a new user."""
    payload = {
        "username": "alice",
        "email": "alice@example.com",
        "password": "Password123!",
        "role": "user"
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["username"] == "alice"
    assert data["role"] == "user"


def test_register_duplicate_username():
    payload = {
        "username": "alice",
        "email": "different_alice@example.com",
        "password": "Password123!"
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 400
    assert "Username already taken" in response.json()["detail"]


def test_register_duplicate_email():
    payload = {
        "username": "alice2",
        "email": "alice@example.com",
        "password": "Password123!"
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]


# =======================================================
# 3. User Login Tests
# =======================================================

def test_login_correct_credentials_json():
    """Case: Login with correct credentials (JSON format)."""
    payload = {
        "username": "alice",
        "password": "Password123!"
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["username"] == "alice"
    assert data["role"] == "user"


def test_login_correct_credentials_form():
    """Case: Login with correct credentials (OAuth2 Form format)."""
    response = client.post(
        "/auth/login",
        data={"username": "alice", "password": "Password123!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_incorrect_password():
    """Case: Login with incorrect password."""
    payload = {
        "username": "alice",
        "password": "WrongPassword999"
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]


def test_login_nonexistent_user():
    payload = {
        "username": "nobody_exists",
        "password": "SomePassword123"
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]


# =======================================================
# 4. Protected API & Token Verification Tests
# =======================================================

def test_access_api_without_token():
    """Case: Access API without a token."""
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_access_api_with_invalid_token():
    """Case: Access API with an invalid token."""
    headers = {"Authorization": "Bearer totally_fake_and_invalid_token_xyz"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 401
    assert "Could not validate credentials" in response.json()["detail"]


def test_access_api_with_valid_token():
    # Login to get token
    login_res = client.post(
        "/auth/login",
        json={"username": "alice", "password": "Password123!"}
    )
    token = login_res.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "alice"
    assert data["email"] == "alice@example.com"
    assert data["role"] == "user"


# =======================================================
# 5. Role-Based Access & Job Authorization Tests
# =======================================================

def test_register_second_user_and_admin():
    # Register bob (user)
    res_bob = client.post("/auth/register", json={
        "username": "bob",
        "email": "bob@example.com",
        "password": "BobPassword123",
        "role": "user"
    })
    assert res_bob.status_code == 201

    # Register admin (admin role)
    res_admin = client.post("/auth/register", json={
        "username": "admin_user",
        "email": "admin@example.com",
        "password": "AdminPassword123",
        "role": "admin"
    })
    assert res_admin.status_code == 201
    assert res_admin.json()["role"] == "admin"


def test_user_tries_to_access_another_user_job():
    """Case: User tries to access another user's job."""
    db = TestingSessionLocal()
    alice = crud.get_user_by_username(db, "alice")
    bob = crud.get_user_by_username(db, "bob")

    # Create a job owned by Alice
    alice_job = crud.create_job(db, job_id="job_alice_01", filename="alice_video.mp4", user_id=alice.id)
    db.close()

    # Bob logs in
    bob_login = client.post("/auth/login", json={"username": "bob", "password": "BobPassword123"})
    bob_token = bob_login.json()["access_token"]
    bob_headers = {"Authorization": f"Bearer {bob_token}"}

    # Bob tries to view Alice's job -> 403 Forbidden
    res_view = client.get(f"/jobs/{alice_job.job_id}", headers=bob_headers)
    assert res_view.status_code == 403
    assert "not authorized" in str(res_view.json()).lower()

    # Bob tries to delete Alice's job -> 403 Forbidden
    res_del = client.delete(f"/jobs/{alice_job.job_id}", headers=bob_headers)
    assert res_del.status_code == 403
    assert "not authorized" in str(res_del.json()).lower()


def test_admin_accesses_all_jobs():
    """Case: Admin accesses all jobs."""
    # Admin logs in
    admin_login = client.post("/auth/login", json={"username": "admin_user", "password": "AdminPassword123"})
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Admin requests all jobs -> Sees jobs from all users
    response = client.get("/jobs/", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["role"] == "admin"
    assert data["total"] >= 1
    job_ids = [j["job_id"] for j in data["jobs"]]
    assert "job_alice_01" in job_ids

    # Admin can view Alice's job detail without 403
    detail_res = client.get("/jobs/job_alice_01", headers=admin_headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["job_id"] == "job_alice_01"


def test_user_can_delete_own_job():
    """A user can successfully delete their own job."""
    alice_login = client.post("/auth/login", json={"username": "alice", "password": "Password123!"})
    alice_token = alice_login.json()["access_token"]
    alice_headers = {"Authorization": f"Bearer {alice_token}"}

    # Alice deletes her own job
    del_res = client.delete("/jobs/job_alice_01", headers=alice_headers)
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Now viewing the job returns 404
    view_res = client.get("/jobs/job_alice_01", headers=alice_headers)
    assert view_res.status_code == 404


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))

