---
description: Code / C++ / misc
---

# mt19937 Random Generator

```cpp
const long long rand_L = 1;
const long long rand_R = 10;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<int> dist(rand_L, rand_R);
auto generator = bind(dist, rng);
```
