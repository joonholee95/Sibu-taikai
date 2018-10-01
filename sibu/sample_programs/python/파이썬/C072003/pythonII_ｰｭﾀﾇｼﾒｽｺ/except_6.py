#Try ~ Finally 문 

import sys
import time

fp = None

try:
	fp = open("test.txt")

	while True:
		line = fp.readline()		
		if len(line) == 0:
			break
		print (line)
		sys.stdout.flush() #print문 뒤에 사용해서 바로바로 화면에 출력하라...
		time.sleep(2)
except IOError:
	print("읽을 파일이 없습니다")
except KeyboardInterrupt:
	print("사용자가 취소를 하였습니다...")
finally:
	if fp:
		fp.close()
	print("파일을 close 처리 했습니다")
     
