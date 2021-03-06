参考链接
•	Java 的 PriorityQueue 文档
•	Java 的 Stack 源码
•	Java 的 Queue 源码
•	Python 的 heapq
•	高性能的 container 库

预习题目
[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

解法：栈
```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        d = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for i in s:
            if i in d:
                stack.append(i)
            else:
                if stack and d[stack[-1]] == i:
                    stack.pop()
                else:
                    return False
         
        return not stack
```
[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

解法：
```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_stack = []


    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)
        if not self.min_stack or x < self.min_stack[-1]:
             self.min_stack.append(x)
        else:
            self.min_stack.append(self.min_stack[-1])


    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop()
        self.min_stack.pop()


    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1] if self.stack else None


    def getMin(self):
        """
        :rtype: int
        """
        return self.min_stack[-1] if self.min_stack else None



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

实战题目
[84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram)

解法：栈
```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack = []
        heights = [0] + heights + [0]
        res = 0
        for i in range(len(heights)):
            #print(stack)
            while stack and heights[stack[-1]] > heights[i]:
                tmp = stack.pop()
                res = max(res, (i - stack[-1] - 1) * heights[tmp])
            stack.append(i)
        return res
```

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum)

解法1：暴力法
```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        n = len(nums)
        if n * k == 0:
            return []
        return [max(nums[i:i + k]) for i in range(n - k + 1)]
```

解法2：双端队列
```python
from collections import deque


class Solution:
    def maxSlidingWindow(self, nums, k):
        # base cases
        n = len(nums)
        if n * k == 0:
            return []
        if k == 1:
            return nums

        def clean_deque(i):
            # remove indexes of elements not from sliding window
            if deq and deq[0] == i - k:
                deq.popleft()

            # remove from deq indexes of all elements
            # which are smaller than current element nums[i]
            while deq and nums[i] > nums[deq[-1]]:
                deq.pop()

        # init deque and output
        deq = deque()
        max_idx = 0
        for i in range(k):
            clean_deque(i)
            deq.append(i)
            # compute max in nums[:k]
            if nums[i] > nums[max_idx]:
                max_idx = i
        output = [nums[max_idx]]

        # build output
        for i in range(k, n):
            clean_deque(i)
            deq.append(i)
            output.append(nums[deq[0]])
        return output

```
课后作业
•	用 add first 或 add last 这套新的 API 改写 Deque 的代码
•	分析 Queue 和 Priority Queue 的源码
[641. 设计循环双端队列](https://leetcode-cn.com/problems/design-circular-deque)

解法：使用python列表实现
```python
class MyCircularDeque(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        :type k: int
        """
        self.deque = []
        self.size = k

    def insertFront(self, value):
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if len(self.deque) == self.size:
            return False
        self.deque.insert(0, value)
        return True

    def insertLast(self, value):
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if len(self.deque) == self.size:
            return False

        self.deque.append(value)
        return True
        

    def deleteFront(self):
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        :rtype: bool
        """
        if not self.deque:
            return False
        del self.deque[0]
        return True

    def deleteLast(self):
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        :rtype: bool
        """
        if not self.deque:
            return False
        del self.deque[-1]
        return True

    def getFront(self):
        """
        Get the front item from the deque.
        :rtype: int
        """
        return self.deque[0] if self.deque else -1

    def getRear(self):
        """
        Get the last item from the deque.
        :rtype: int
        """
        return self.deque[-1] if self.deque else -1

    def isEmpty(self):
        """
        Checks whether the circular deque is empty or not.
        :rtype: bool
        """
        return not self.deque

    def isFull(self):
        """
        Checks whether the circular deque is full or not.
        :rtype: bool
        """
        return len(self.deque) == self.size


# Your MyCircularDeque object will be instantiated and called as such:
# obj = MyCircularDeque(k)
# param_1 = obj.insertFront(value)
# param_2 = obj.insertLast(value)
# param_3 = obj.deleteFront()
# param_4 = obj.deleteLast()
# param_5 = obj.getFront()
# param_6 = obj.getRear()
# param_7 = obj.isEmpty()
# param_8 = obj.isFull()
```
[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)
