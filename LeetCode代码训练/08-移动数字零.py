class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        def move(x:int):
            """
            将x位上的数移动到最近的一个合适0之前位置上。
            """
            n = len(nums)-1
            end = n
            #range(n,-1,-1),包头不包尾，只能到0.
            for i in range(n,-1,-1):
                if nums[i]!=0:
                    end = i
                    break
            while x < end:
                nums[x],nums[x+1]=nums[x+1],nums[x]
                x+=1
                
        if nums[0]==0:
            move(0)
            
        for i in range(1,len(nums)):
            if nums[i-1]==0:
                move(i-1)
            elif nums[i]==0:
                move(i)
        return nums

#大神算法
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = 0
        j = 0
        for i in range(len(nums)):
            if nums[i]!=0:
                nums[j],nums[i]=nums[i],nums[j]#j代表前j位非零元素，零值会被j值逐渐过渡取代掉
                j+=1
        return nums
