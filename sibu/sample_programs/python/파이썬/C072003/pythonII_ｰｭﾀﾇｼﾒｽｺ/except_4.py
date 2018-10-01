# 예외 발생시키기 : raise 명령어를 이용해서 에러를 강제로 발생시키는 방법
class Fleight:
	def fly(self):
		raise NotImplementedError # 파이썬의 내장 에러로 구현되지 않았을 때 발생시키는 에러


class Plane(Fleight):
	#pass
	def fly(self):
		print("빠른 속도로 날아가는 비행기 입니다")


plane = Plane()
plane.fly()