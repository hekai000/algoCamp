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
4. [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

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

三数之和题解：

解法1：暴力求解

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        length = len(nums)
        if length < 3:
            return []
        nums.sort()
        result = []
        for i in range(length -  2):
            for j in range(i + 1, length - 1):
                for k in range(j + 1, length):
                    if nums[i] + nums[j] + nums[k] == 0 and [nums[i], nums[j], nums[k]] not in result:
                        result.append([nums[i], nums[j], nums[k]])
        return result
```

解法2：
排序 + 哈希

先对数组进行排序(理由稍后解释), 再对排序后的数组进行遍历, 将每个元素的相反数作为key, 元素所在的位置作为value存入哈希表中, 两次遍历数组不断检查 a + b 之和是否存在于哈希表中.

有以下几个需要注意的点:
(1)找到满足条件的结果后, 需要将结果数组序列化并存入令一个哈希表中, 以便对结果去重
(2)首先在对 a,b 进行遍历时, 如果当前元素与前一个元素相同可直接跳过以优化性能 (思考: 后一个元素能发现的结果一定会包含在前一个元素的结果中). 
另外, 仅在一层循环中加入此逻辑性能最佳. 该逻辑有效的前提是相同的元素需要连在一起, 所以需先对数组进行排序。

有个疑问：为什么i + j + 1 == target_hash[x+y]不行?

复杂度评估
时间复杂度 O(n ^ 2)
空间复杂度 O(n)
```python
class Solution(object):
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        res = []
        res_hash = {}
        nums.sort()
        target_hash = {-x: i for i, x in enumerate(nums)}

        for i, x in enumerate(nums):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            for j, y in enumerate(nums[i+1:]):
                if x + y in target_hash:
                    if i == target_hash[x+y] or j + 1 + i == target_hash[x+y]:
                        continue
                    row = sorted([x, y, -x-y])
                    key = ','.join([str(item) for item in row])
                    if key not in res_hash:
                        res_hash[key] = row
                        res.append(row)

        return res
```

解法3：

排序 + 双指针
本题的难点在于如何去除重复解。

算法流程：
特判，对于数组长度n，如果数组为null 或者数组长度小于 3，返回[]。
对数组进行排序。
遍历排序后数组：
若 nums[i]>0：因为已经排序好，所以后面不可能有三个数加和等于0，直接返回结果。
对于重复元素：跳过，避免出现重复解
令左指针 L=i+1，右指针 R=n-1，当 L<R时，执行循环：
当 nums[i]+nums[L]+nums[R]==0，执行循环，判断左界和右界是否和下一位置重复，去除重复解。并同时将 L,R移到下一位置，寻找新的解
若和大于 0，说明 nums[R]太大，R左移
若和小于0，说明 nums[L]太小，L右移

复杂度分析
时间复杂度：O(n^2)，数组排序 O(NlogN)，遍历数组 O(n)，双指针遍历 O(n)，总体 O(NlogN)+O(n)∗O(n)，O(n^2)
空间复杂度：O(1)

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        length = len(nums)
        nums.sort()
        res = []
        for i in range(length):
            if nums[i] > 0:
                return res
            if i > 0  and nums[i] == nums[i - 1]:
                continue
            j = i + 1
            k = length - 1
            while j < k:
                if nums[j] + nums[k] < -nums[i]:
                    j += 1
                elif nums[j] + nums[k] > -nums[i]:
                    k -= 1
                else:
                    res.append([nums[i], nums[j], nums[k]])
                    while j < k and nums[j] == nums[j + 1]:
                        j += 1
                    while j < k and nums[k] == nums[k - 1]:
                        k -= 1
                    j += 1
                    k -= 1

        return res
```


Linked List 实战题目
[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None
        cur = head
        
        while cur:
            post = cur.next
            cur.next = pre
            pre = cur
            cur = post
        return pre
```

[24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs)

[141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle)

解法1：快慢指针
```python
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        return True
```

解法2：哈希表
```python
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return False
        d = {}
        while head:
            if head in d:
                return True
            else:
                d[head] = 1
            head = head.next
        return False
```
[142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii)

解法1：哈希
```python

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        node_set  = set()
        while(head):
            if head in node_set:
                return head
            node_set.add(head)
            head = head.next
        return None
```

解法2：快慢指针，如何理解从相遇点和头结点同时出发再次相遇的点一定是环的入口节点。
解题思路：
这类链表题目一般都是使用双指针法解决的，例如寻找距离尾部第K个节点、寻找环入口、寻找公共尾部入口等。

算法流程：
双指针第一次相遇： 设两指针 fast，slow 指向链表头部 head，fast 每轮走2步，slow 每轮走1步；

第一种结果： fast指针走过链表末端，说明链表无环，直接返回 null；

TIPS: 若有环，两指针一定会相遇。因为每走1轮，fast与slow的间距 +1，fast终会追上slow；
第二种结果： 当fast == slow时， 两指针在环中第一次相遇 。下面分析此时fast 与slow走过的步数关系：

设链表共有 a+b个节点，其中 链表头部到链表入口 有 a 个节点（不计链表入口节点）， 链表环 有 b个节点（这里需要注意，a 和 b 是未知数，例如图解上链表 a=4, b=5）；
设两指针分别走了 f，s步，则有：
fast 走的步数是slow步数的 2倍，即 f = 2s；（解析： fast 每轮走 2 步）
fast 比 slow多走了 n 个环的长度，即 f = s + nb；（ 解析： 双指针都走过 a 步，然后在环内绕圈直到重合，重合时 fast 比 slow 多走环的长度整数倍 ）；
以上两式相减得：f = 2nb，s = nb，即fast和slow 指针分别走了 2n，n个环的周长 （注意： n是未知数，不同链表的情况不同）。

目前情况分析：
如果让指针从链表头部一直向前走并统计步数k，那么所有走到链表入口节点时的步数是：k=a+nb（先走 a步到入口节点，之后每绕 1圈环（ b步）都会再次到入口节点）。
而目前，slow 指针走过的步数为 nb步。因此，我们只要想办法让 slow 再走 a步停下来，就可以到环的入口。
但是我们不知道 a的值，该怎么办？依然是使用双指针法。我们构建一个指针，此指针需要有以下性质：此指针和slow 一起向前走 a 步后，两者在入口节点重合。
那么从哪里走到入口节点需要 a步？答案是链表头部head。
双指针第二次相遇：

slow指针 位置不变 ，将fast指针重新 指向链表头部节点 ；slow和fast同时每轮向前走 1步；
TIPS：此时 f = 0，s=nb ；
当 fast 指针走到f = a 步时，slow 指针走到步s = a+nb，此时 两指针重合，并同时指向链表环入口 。
返回slow指针指向的节点。

复杂度分析：
时间复杂度 O(N) ：第二次相遇中，慢指针须走步数 a < a + b；第一次相遇中，慢指针须走步数 a + b - x < a + b，其中 x为双指针重合点与环入口距离；因此总体为线性复杂度；
空间复杂度 O(1) ：双指针使用常数大小的额外空间。


```python
class Solution(object):
    def getIntersection(self, head):
        fast = head
        slow = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return slow
        return None

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        intersection = self.getIntersection(head)
        if not intersection:
            return None
            
        ptr1  = head
        ptr2 = intersection
        while ptr1 != ptr2:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        return ptr1
```
[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

课后作业
[26. 删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)
[189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

解法1：切片
```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        if not n:
            return
        nums[:] = nums[n-k:] + nums[:n-k]
```

解法2：会超时, O(n^2)

```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.

        """
        n = len(nums)
        k %= n
        while k:
            tmp = nums[-1]
            for i in range(len(nums) - 1, 0, -1):
                nums[i] = nums[i - 1]
            nums[0] = tmp
            k -= 1
```

解法3：三次反转

```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.

        """
        n = len(nums)
        k %= n
        nums[:] = nums[::-1]
        nums[:k] = nums[:k][::-1]
        nums[k:] = nums[k:][::-1]
```

解法4： 插入

```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.

        """
        n = len(nums)
        k %= n
        for _ in range(k):
            nums.insert(0, nums.pop())
```
[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

解法1： 递归(最近重复子问题)
```python
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 or not l2:
            return l1 or l2
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

解法2：迭代
```python
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 or not l2:
            return l1 or l2
        head = dummy = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                head.next = l1
                head = l1
                l1 = l1.next
            else:
                head.next = l2
                head = l2
                l2 = l2.next
        if l1 or l2:
            head.next = l1 or l2
        return dummy.next
        
```
[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

解法1：合并后排序
```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        nums1[:] = sorted(nums1[:m] + nums2)
```

解法2：双指针(从前往后)
```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        nums1_copy = nums1[:m]
        nums1[:] = []
        i = 0
        j  = 0
        while i < m and j < n:
            if nums1_copy[i] < nums2[j]:
                nums1.append(nums1_copy[i])
                i += 1
            else:
                nums1.append(nums2[j])
                j += 1
        if i < m:
            nums1[i+j:] = nums1_copy[i:]
        if j < n:
            nums1[i+j:] = nums2[j:]
        return
```

解法3：双指针(从后往前)
```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        p1 = m - 1
        p2 = n - 1
        p = m + n - 1
        while  p1 >=0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] = nums1[p1]
                p1 -= 1
            p -= 1
        nums1[:p2 + 1] = nums2[:p2 + 1]
```
[1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = {}
        for i in range(len(nums)):
            if nums[i] in d:
                return [d[nums[i]], i]
            else:
                d[target - nums[i]] = i
```
[283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)
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
                if i != j:
                    nums[i] = 0
                j += 1
        return
```
[66. 加一](https://leetcode-cn.com/problems/plus-one/)
先将列表转化为数字，再加一，再将数字转化为列表。
```python
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        sum = 0
        for i in range(len(digits)):
            sum = 10 * sum + digits[i]
        new_sum = sum + 1
        return [int(item) for item in list(str(new_sum))]
```