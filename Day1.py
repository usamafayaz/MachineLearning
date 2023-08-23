import random as rd
import math

# 1. Prime Number Detector


# 2. Program the counts the iteration of 10 '6' or total '10' point of dice.

# totalReps=0
# counter=0
# while True:
#     totalReps=totalReps+1
#     dice1 = rd.randint(1, 6)
#     dice2 = rd.randint(1, 6)
#     if dice1==6 or dice1+dice2>=10 or dice2==6:
#         counter+=1
#     if counter ==10:
#         break
#
# print("Total Reps are :", totalReps)

# 3. Dictionary in Python.

# students ={
#     'arid_1':'Usama',
#     'arid_2':'Anees',
#     'arid_3':'Abdullah'
# }

# for key,val in students.items():
#     print("Key is :",key)
#     print("Value is: ",val)
#
# print(students.items())

# 4. Program to make a list of students that have CGPA greater than 3.5

# students ={
#     'arid_1':['Usama',3.88],
#     'arid_2':['Zia',3.87],
#     'arid_3':['Abdullah',3.56],
#     'arid_4':['Anees',3.0]
# }
# topper=[]
# for k,v in students.items():
#     if v[1]>=3.5:
#         topper.append(k)
#
# print(topper)

# 4. CRUD Operation in Dictionary.

# def DisplayData(printSt):
#     print(printSt)
#     for key,val in dict.items():
#         print("Key--> ", key, "Value--> ",val)
#
# dict ={
#     'k1':'v1',
#     'k2':'v2',
#     'k3':'v3',
# }
# DisplayData("Dictionary At Start.")
#
# dict.pop('k2')
# DisplayData("Dictionary after Deleting.")
#
# dict['k4']='v4'
# DisplayData("Dictionary after Adding.")
#
# dict.update({'k4':'v8'})
# DisplayData("Dictionary after Updating.")

# 4. Program to make a list of students which are dropped by
#    comparing there CGPA.

students ={
    'arid_1':['Usama',0.75,8],
    'arid_2':['Zia',3.87,5],
    'arid_3':['Abdullah',1.75,6],
    'arid_4':['Anees',3.0,6],
    'arid_5':['Adeel',2.2,7],
}
dropStudents={}
dropLst=[0.75,1.25,1.75,2.25]
for k,v in students.items():
    if v[1]<dropLst[math.ceil((v[2]/2)-1)]:

        dropStudents[k]=v

print("List of Dropped Students:")
for k,v in dropStudents.items():
    print("We are sorry",v[0],"! Minimum requirement for you was ",dropLst[math.ceil((v[2]/2)-1)])