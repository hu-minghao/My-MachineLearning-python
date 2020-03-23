#我的方法，耗时少，但内存占用大。
#加一算法比较简单，可以从算法具体步骤上进行求解。
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)
        old=0
        for i in range(n):
            old += digits[i]*10**(n-i-1)
        new=old+1
        new_digits=[int(x) for x in list(str(new))]
        return new_digits
       
