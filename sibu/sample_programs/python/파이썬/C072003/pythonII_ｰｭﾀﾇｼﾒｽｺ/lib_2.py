#OS모듈
"""
	os.environ : 시스템의 환경 변수값 들을 보여주는 역할을 한다.
					 시스템의 정보를 딕셔너리 객체로 돌려준다.
    os.chdir : 현재 디렉토리의 위치를 변경하는 함수
	os.getcwd : 자신의 현재 디렉토리의 위치를 돌려준다.

	os.system("명령어") :시스템의 유틸리티나 dos 명령어들을 파이썬에서 호출 한다.
	os.popen : 시스템 명령어를 시킨 결과값을 읽기모드의 파일 객체로 돌려준다.
    
	* 기타 os모듈의 유용한 함수들
	 os.mkdir(디렉토리명) : 디렉토리를 생성한다.
	 os.rmdir(디레토리명) : 디렉토리를 삭제한다. (디렉토리가 비어있어야 한다.)
	 os.unlink(파일명) : 파일을 지운다.
	 os.rename(src, dst) :src이름파일을 dst이름으로 바꾼다. 
"""

# shutil : 파일을 복사해 주는 모듈
"""
shutil.copy(src, dst) : src라는 이름으로 파일을 dst로 복사한다.

"""
# glob 모듈: 디렉토리에 있는 파일들을 리스트로 만들 때 사용한다.









import os
os.environ