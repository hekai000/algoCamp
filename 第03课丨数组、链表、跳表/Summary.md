# 数组、链表、跳表

## 数组、链表实现、复杂度

数据结构 | 查找 | 插入| 删除
--- |---   | ---  | ---|
数组 | O(1) | O(n) | O(n)
链表 | O(n) | O(1) | O(1)

## LRU Cache

## 跳表

### 跳表原理

空间换时间

### Redis为什么用跳表，不用红黑树？

1. 在做范围查找的时候，平衡树比skiplist操作要复杂。在平衡树上，我们找到指定范围的小值之后，还需要以中序遍历的顺序继续寻找其它不超过大值的节点。如果不对平衡树进行一定的改造，这里的中序遍历并不容易实现。而在skiplist上进行范围查找就非常简单，只需要在找到小值之后，对第1层链表进行若干步的遍历就可以实现。
平衡树的插入和删除操作可能引发子树的调整，逻辑复杂，而skiplist的插入和删除只需要修改相邻节点的指针，操作简单又快速。

2. 从内存占用上来说，skiplist比平衡树更灵活一些。一般来说，平衡树每个节点包含2个指针（分别指向左右子树），而skiplist每个节点包含的指针数目平均为1/(1-p)，具体取决于参数p的大小。如果像Redis里的实现一样，取p=1/4，那么平均每个节点包含1.33个指针，比平衡树更有优势。
查找单个key，skiplist和平衡树的时间复杂度都为O(log n)，大体相当；而哈希表在保持较低的哈希值冲突概率的前提下，查找时间复杂度接近O(1)，性能更高一些。所以我们平常使用的各种Map或dictionary结构，大都是基于哈希表实现的。

3. 从算法实现难度上来比较，skiplist比平衡树要简单得多。

## 实战题目

### Array 实战题目
1. [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/) 
2. [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/) 
3. [70. 爬楼梯](https://leetcode.com/problems/climbing-stairs/)
4. [link](https://leetcode-cn.com/problems/3sum/)

盛最多水的容器题解：

解法一：
两层遍历
```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        max_area = 0
        for i in range(len(height) - 1):
            for j in range(i + 1, len(height)):
                max_area = max(max_area, (j - i) * min(height[i], height[j]))
        return max_area
                
```

解法二：
```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height) - 1
        max_area = 0
        while left < right:
            if height[left] < height[right]:
                area = height[left] * (right - left)
                left += 1
            else:
                area = height[right] * (right - left)
                right -= 1
            max_area = max(max_area, area)
        return max_area
```

移动零题解：

解法一：

先将非零调整到前面的位置(其中j用来记第一个0的位置，i用来遍历数组), 再对数组后面的元素赋零
```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1
        while j < len(nums):
            nums[j] = 0
            j += 1
        return
```

解法2：

在调整非零元素到前面位置的同时，进行后面元素的赋零操作
```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                if j != i:
                    nums[i] = 0
                j += 1
        return
```

解法3：

碰到非零元素即交换
```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1
        return
```

爬楼梯题解：
找最近重复子问题。
n = 1, 一种跳法；
n = 2, 两种跳法；
n = 3, 要么从n=1跳两个台阶，要么从n=2跳一个台阶，即f(n) = f(n-1) + f(n-2)，与斐波那契数列解法一致。
```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n

        f1, f2, f3 = 1, 2, 3
        for i in range(3, n + 1):
            f3 = f1 + f2
            f1, f2 = f2, f3
        return f3
```