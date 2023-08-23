from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
# Create a database engine

engine = create_engine('sqlite:///mydatabase.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
def returnSession():
# Create a session class
    return session

def updateTables(Base):
     Base.metadata.create_all(engine)


Base= declarative_base()