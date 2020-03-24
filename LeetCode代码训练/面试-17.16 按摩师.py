class Solution:
    def massage(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 2)
        ans = 0
        for i in range(2, n + 2):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 2])
            ans = max(ans, dp[i])
        return ans
        #动态规划，如果接受当前状态，获利是d[i-2],不接受则获利是d[i-1]
        #使用dp[]存储获利状态，将最大获利状态存在当前位置
