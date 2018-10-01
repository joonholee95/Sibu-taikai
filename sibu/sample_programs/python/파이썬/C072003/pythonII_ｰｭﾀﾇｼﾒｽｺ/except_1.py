'''
 [ 예외처리 ]
	
	예외: 프로그램에서 벌어지는 예외적인 상황(아래 에러들)을 의미
	예> 파일을 읽고자 할 때 그 파일이 존재하지 않을 경우
	      프로그램이 한참 실행중일 때 갑자기 그파일이 지워진 경우
     
		이러한 예외 상황을 처리해 주는 것을 예외 처리라고 한다.

		에러의 예>
			FileNotFoundError(파일이 없을 때) 
			ZeroDivisionError(숫자를 0으로 나누었을때)
			IndexError(리스트에서 얻어 올 수 없는 값일 경우)
			SyntaxError(구문 오류)
			EOFError(파일의 끝일 경우:읽을 내용이 없을 때)
			.....

		파이썬은 이러한 에러들이 발생하면 프로그램을 중단하고 에러 메시지를 보여준다.

		파이썬의 예외처리 기본 형식(자바의 try ~ catch문)
        
			try:
				수행 명령......
			except [발생에러[as 에러메시지 변수]]:
				수행 명령......

        1)                    2)                             3)
		 try:                 try:                           try:
			 ....                 ......                          .....
         except:            except 발생에러:          except 발생에러 as 에러메시지 변수: 
			 ....                 ......                          .....

'''

try:
	str = input("문자를 입력하세요 >> ")
except EOFError:
	print ("읽을 내용이 없습니다!!!")
except KeyboardInterrupt:
	print("입력 취소되었습니다...!!!")
else:
	print ("당신이 입력한 문자 --> {} ".format(str))

