# Engine
# Session Manager
# Base Class

from flask import Flask, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sqlalchemy as db

from sqlalchemy.ext.declarative import declarative_base

# Create a Flask app
app = Flask(__name__)

# Create a SQLite database engine
# connects to the db and do processing
engine = create_engine("sqlite:///studentdb.db", echo=True)

# Create a session factory
# # handle data in memory in your project
# DB ka data memory mein save

Session = sessionmaker(bind=engine)
session = Session()

# Create a base class for declarative models
Base = declarative_base()


# Define your models here
class Student(Base):
    __tablename__ = 'students'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)


# Create the database tables
Base.metadata.create_all(engine)

newStudent=Student(name="Sarmad",age=20)
session.add(newStudent)
session.commit()

# Route to get student data
@app.route('/students', methods=['GET'])
def get_students():
    print('bolo')
    students = session.query(Student).all()
    student_list = [{'id': student.id, 'name': student.name, 'age': student.age} for student in students]
    return jsonify(student_list)
@app.route('/clear_students', methods=['POST'])
def clear_students():
    try:
        # Clear the students table
        session.query(Student).delete()
        session.commit()
        return jsonify({"message": "Table cleared successfully"})
    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
