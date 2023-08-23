# OOP in Python
# Class in Python

# class Student:
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#
# std = Student("Usama",23)
# print(std.name)
# print(std.age)


# 2. Bank Scenario

# class Bank:
#     def __init__(self,AccNumber,name,amount):
#         self.AccNumber=AccNumber
#         self.amount=amount
#         self.name=name
#     def deposit(self, amount):
#         self.amount+=amount
#     def withdraw(self, amount):
#         self.amount-=amount
#     def showBalance(self):
#         print(self.name,"balance is ",self.amount)
#     def transfer(self,newAmt,acct):
#         self.withdraw(newAmt)
#         acct.deposit(newAmt)
#
# acc1 = Bank(123,"Usama",20000)
# acc2 = Bank(124,"Abdullah",20000)
# acc1.showBalance()
# acc2.showBalance()
#
# acc2.transfer(10000,acc1)
#
# acc1.showBalance()
# acc2.showBalance()

# 3. Inheritance in Python

# class Person:
#     def __init__(self, name, age):
#         self.name=name
#         self.age=age
#
#     def printCredentials(self):
#         print("Parent Call !!",self.name ," ", self.age)
#
#
# class Student(Person):
#     def __init__(self, name, age, aridNo):
#         super().__init__(name,age)
#         self.aridNo=aridNo
#     def printData(self):
#         super().printCredentials()
#         print(self.aridNo)
#
# class Employee(Person):
#     def __init__(self, name, age, empNo, Department):
#         super().__init__(name,age)
#         self.empNo=empNo
#         self.Department=Department
#
#     def printData(self):
#         super().printCredentials()
#         print(self.empNo, " ", self.Department)
#
#
# std= Student('Usama',23,4232)
# std.printData()
#
# emp=Employee("Abdullah",24,1,"CS")
# emp.printData()
#
import matplotlib.pyplot as pt
import numpy as np

# 4. Matplotlib is used to plot the graphs.

# x_data=np.array(['UnEducated','Matric','Intermediate','Above'])
# y_data=np.array([30,40,25,5])
#
# pt.bar(x_data,y_data)
# pt.xlabel('Level')
# pt.ylabel('Percentage')
# pt.grid()
# pt.show()

# mylabels=['UnEducated','Matric','Intermediate','Above']
# myexplode = [0, 0, 0, 0.2]
# pt.pie(y_data, labels = mylabels, explode = myexplode, autopct='%1.1f%%')

#pt.legend(['UnEducated','Matric','Intermediate','Above'])
# pt.show()

# 5. Pakistan and India Score

# import numpy as np
# import matplotlib.pyplot as plt
#
# Pakistan_Score = np.array([40, 55, 85, 180])
# India_Score = np.array([50, 95, 125, 160])
# Overs = [5, 10, 15, 20]
#
# plt.plot(Overs, Pakistan_Score, 'g', label='Pakistan')
#
# plt.plot(Overs, India_Score, 'b--', label='India')
#
# plt.xlabel('Overs')
# plt.ylabel('Score')
# plt.title('Pakistan vs India Cricket Match')
# #plt.legend()
#
# plt.show()


# 6. SubPlot

# import numpy as np
# import matplotlib.pyplot as plt
#
# Pakistan_Score = np.array([40, 55, 85, 180])
# India_Score = np.array([50, 95, 125, 160])
# Overs = [5, 10, 15, 20]
#
# plt.subplot(1,2,2)
# plt.plot(Overs, Pakistan_Score, 'g', label='Pakistan')
#
# plt.subplot(1,2,1)
# plt.plot(Overs, India_Score, 'b--', label='India')
#
# plt.xlabel('Overs')
# plt.ylabel('Score')
# plt.title('Pakistan vs India Cricket Match')
# plt.legend()
#
# plt.show()

# 7. Pandas can read any type of file and convert it into dataset
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('IndiaCovidCasesByStateUT.csv')
# print(df.head(10))
# print(df.tail())
# print(df.shape) # prints rows and columns

states=np.array(df.get('State/UTs'))
cases=np.array(df.get('Total Cases'))

plt.xlabel('States')
plt.ylabel('Cases')
plt.title('Corona in India')
plt.xticks(rotation=40,ha='right')

bars = plt.bar(states, cases)
bars[-1].set_color('red')

plt.bar(states[::-1],cases[::-1])
plt.show()
#
#
#
