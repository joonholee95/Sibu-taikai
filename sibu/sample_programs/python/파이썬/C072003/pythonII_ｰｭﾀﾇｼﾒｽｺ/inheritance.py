""" 상속 : 재사용의 한가지 방법
객체지향의 가장 큰장점은 코드의 재사용(재활용)
Person 클래스를 상속받는 Student 클래스를 만들때 
표현방법은 Student(Person)
이때 Person 클래스를 슈퍼클래스라고 하고, Student 클래스를 서브클래스(하위클래스)
슈퍼클래스는 부모클래스, 서브클래스는 자식클래스라고도 한다.

"""

class Person:
	def __init__(self, name, age):
		self.name = name
		self.age = age
		print ("{} 객체를 만드는 중".format(self.name))
	
	def speak(self):
		print("내이름은 '{}' 나이는 '{}'".format(self.name, self.age))

class Student(Person):
	def __init__(self, name, age, hakbun):
		Person.__init__(self,name,age)
		self.hakbun = hakbun
		print ("{} 학생 객체를 만드는 중....".format(self.name))
	def speak(self):
		Person.speak(self)
		print ("나는 {:d} 학번 입니다".format(self.hakbun))

class Professor(Person):
	def __init__(self, name, age, pay):
		Person.__init__(self, name, age)
		self.pay = pay
		print("{} 교수객체를 만드는 중...".format(self.name))

	def speak(self):
		Person.speak(self)
		print ("페이가 {:d} 인 교수".format(self.pay))

s = Student("홍길동", 15, 20150303)

p = Professor("김교수", 38, 3000)

member = [s, p]
for aa in member:
	aa.speak()
