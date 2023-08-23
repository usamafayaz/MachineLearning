# difference between DB and ML Model
# Ability to perform on Unseen data.
# Flask Server have both of them i.e. DB+ML Model

# Controller Breaks down the heirarchy according to the application data flow.
# Controller contains bussiness logic implementation.

# Model is used to store the relational data of DB.

# View is not necessary for us, we will use react to render the data to user.

# SQLAlchemy ORM just as LINQ -> Object Relational Mapper

# select * from student
# db.query().all()  // Functional Approach

# *Code First // Model is already created, then DB is created automaticlly
# Data First  // DB is already created, then model is created automaticlly

# 1. flask Library

# from flask import Flask,jsonify
#
# app=Flask(__name__)
# @app.route('/Student/<string:name>')
# def Welcome(name):
#     return "Hello "+name
#
# if __name__=='__main__':
#     app.run()

#  2. Student Fetch.
from flask import Flask,jsonify

app=Flask(__name__)
student = [
    {
        "arid":"2020-arid-4232",
        "name":"Usama",
        "cgpa":3.88,
        "class":"BSCS-6D",
    },
    {
        "arid":"2020-arid-3588",
        "name":"Abdullah",
        "cgpa":3.60,
        "class":"BSCS-6D",
    },
    {
        "arid":"2020-arid-3694",
        "name":"Anees",
        "cgpa":3.50,
        "class":"BSCS-6D",
    }
]

@app.route('/allstudents')
def FetchAll():
    return jsonify(student)

@app.route('/fetchstudent/<string:arid>')
def fetchstudent(arid):
    for i in student:
        if i["arid"]==arid:
            return jsonify(i)
    return jsonify({"Message":"No Student Found!"})


if __name__=='__main__':
    app.run()
