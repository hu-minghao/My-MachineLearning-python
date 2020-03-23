class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        carry = 1
        for i in range(len(digits)-1, -1, -1):
            if digits[i]==9:
                if carry==1:
                    digits[i] = 0
                    carry = 1
            else:
                digits[i] += carry
                carry = 0
        if carry==1: return [1]+digits
        else: return digits

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)-1
        while True:
            if digits[n]==9 and n==0:
                digits[n] = 0
                digits.insert(0,1)
                break
            elif digits[n]==9:
                digits[n]=0
                n-=1
            else:
                digits[n] += 1
                break
        return digits
