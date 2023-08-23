from sqlalchemy.orm import declarative_base
import sqlalchemy as db
import Database as dt

Base=declarative_base()

class Student(Base):
    _tablename_ = 'Student'
    id=db.Column(db.Integer, primary_key=True)
    name=db.Column(db.String)
    section=db.Column(db.String)
    arid_number=db.Column(db.String)

session=dt.returnSession()
S=[Student(name="Hassan",section="BSC-7",arid_number="Arid-1"),
Student(name="Waseem",section="BSC-7",arid_number="Arid-2")

   ]

dt.updateTables(Base)
session.add_all(S)
session.commit()