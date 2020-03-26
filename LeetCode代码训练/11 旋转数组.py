class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        '''
        将行变为列
        然后列反向重排
        '''
        n=len(matrix)
        for i in range(n):
            for j in range(i,n):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
        for i in matrix:
            i.reverse()  
            
 #其中我使用while循环，会报下标溢出的错误。
 for k in range(n):
    l=0
    r=n-1
    while l!=r:
      matrix[k][l],matrix[k][r]=matrix[k][r],matrix[k][l]
      l+=1
      r-=1
#不知道问题在哪。
