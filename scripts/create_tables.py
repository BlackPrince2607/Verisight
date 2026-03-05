from apps.api.db.session import engine
from apps.api.models.models import Base

Base.metadata.create_all(bind=engine)

print("✅ Tables created successfully")
