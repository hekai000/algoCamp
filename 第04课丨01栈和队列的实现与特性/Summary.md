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
[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum)
课后作业
•	用 add first 或 add last 这套新的 API 改写 Deque 的代码
•	分析 Queue 和 Priority Queue 的源码
[641. 设计循环双端队列](https://leetcode-cn.com/problems/design-circular-deque)
[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)
