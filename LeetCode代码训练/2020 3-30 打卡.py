0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

# 我的思路，每次前进m步，从0开始，当然会超过数组长度，这时候取余。位置参数需要-1
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        i, a = 0, list(range(n))
        while len(a)>1:
            i = (i+m-1)%len(a)
            a.pop(i)
        return a[0]
    #上述是别人的代码，但思路相同，虽然我的报错了
#递归，递归就太厉害了
sys.setrecursionlimit(100000)

class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        return self.f(n, m)

    def f(self, n, m):
        if n == 0:
            return 0
        x = self.f(n - 1, m)
        return (m + x) % n

#感觉没有直接求方便
