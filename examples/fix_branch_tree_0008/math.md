A tree is a **recursively defined** object — a node joining subtrees that are themselves trees:

$$\mathrm{Tree} = \mathrm{Node}(\underbrace{\mathrm{Tree}, \dots, \mathrm{Tree}}_{b}).$$

This equation is a fixed point of the branching functor: the type on the left appears unchanged on the right. Unrolling it gives self-similarity at every scale, the natural-world face of $f(x)=x$.