class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        ans = True
        def check(item):
            """
            检查是否有重复的数字，有返回True,没有返回False.
            """
            flag = False
            lis=["1","2","3","4","5","6","7","8","9"]
            for i in item:
                if (i!='.') and (i in lis):
                    lis.remove(i)
                elif (i!='.') and (i not in lis):
                    flag = True
            return flag
            
        for i in range(9):
            if check(board[i]):
                ans = False
                break
            else:
                chid_row = []
                for j in range(9):
                    chid_row.append(board[j][i])
                if check(chid_row):
                    ans = False
        for row in range(3):#9块
            for column in range(3):
                tmp = [board[i][j] for i in range(row*3, row*3+3) for j in range(column*3, column*3+3)]
                if check(tmp):
                    ans = False
        return ans
      # 构造检查函数，然后对每行，每列，每块进行检查
