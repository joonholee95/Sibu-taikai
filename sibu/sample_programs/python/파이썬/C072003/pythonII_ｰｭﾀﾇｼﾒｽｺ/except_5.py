# 사용자 정의 예외 만들어서 발생시키기
#에러(오류)나 예외는 반드시 직접적이든 간접적이든 Exception 클래스에서 파생된 클래스이어야 한다.


class UserException(Exception): #사용자가 정의하는 예외클래스
	def __init__(self, length, minimum):
		Exception.__init__(self)
		self.length = length
		self.minimum = minimum


try:
	txt = input("입력 내용 >> ")
	if len(txt) < 5:
		raise UserException(len(txt), 5)
except EOFError:
	print ("읽을 내용이 없습니다.")
except UserException as uex:
	print(("UserException : 입력된 내용은 문자열의 길이가 {0} 입니다." + \
	"최소한 길이가 {1}이어야 합니다").format(uex.length, uex.minimum))
else:
	print("예외상항이 발생하지 않았습니다...!!!")