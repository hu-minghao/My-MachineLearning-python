#报错，说不存在dic.has_key()方法

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic={}
        res=[]
        for i,j in enumerate(nums):
            dic[i]=j
        for j in dic.keys():
            if 	dic.has_key(target-j):
                res.append(dic[j])
                res.append(dic[target-j])
        return res
#解决了重复的问题
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dct = {}
        for i, n in enumerate(nums):
            if target - n in dct:
                return [dct[target - n], i]
            dct[n] = i
