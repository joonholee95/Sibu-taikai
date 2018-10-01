# 클래스 변수와 객체변수의 예

class Man:
	# 클래스 변수
	cnt = 0

	def __init__(self, name): #__init__메소드를 생성자라고도 한다.
		""" 데이터 초기화 하기 """
		self.name = name #self.name에서 name은 객체변수를 의미한다.
		print("({}이(가) 만들어지는 중.....)".format(self.name))

		Man.cnt +=1 #클래수 변수 접근하기 : 클래스명.클래스변수명
   
	def die(self):
		"""Man 객체가 죽었을 때"""
		print("{} 는 죽었습니다!!!".format(self.name))

		Man.cnt -=1

		if Man.cnt == 0:
			print ("{} 는 최후의 생존자 였다".format(self.name))
		else:
			print("아직 {:d}명의 생존자가 남아있다".format(Man.cnt))
	
	def say(self):
		print ("생성완료!!!!   안녕하세요!!! 내이름은 {} 입니다".format(self.name))

	@classmethod #장식자(decorator) : how_many = classmethod(how_many)의 다른 표현
	def how_many(how):
		""" 현재의 객체 수량을 체크한다"""
		print("현재 {} 명이 남았습니다".format(Man.cnt))
	
	#how_many = classmethod(how_many)
	

gameActor1 = Man("맨1")
gameActor1.say()

gameActor2 = Man("맨2")
gameActor2.say()

gameActor3 = Man("맨3")



print("------------------------------")
gameActor2.die()
Man.how_many() #클래스 메소드 호출 방법: 클래스명.메소드명

