---
description: Code / C++ / Data Structure / External Library / Ordered(Indexed) set
---

# Ordered Set

```cpp
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using ordered_set = tree<int, null_type, less<int>, rb_tree_tag,tree_order_statistics_node_update>;
// order_of_key()
// find_by_order()
```
