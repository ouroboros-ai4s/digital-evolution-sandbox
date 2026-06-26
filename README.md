# digital-evolution-sandbox

🚧 Work in progress

A Red Queen-style symmetric melee evolution sandbox: virtual species compete, mutate, and antagonize across a 128² grid world, with every tick of the whole run faithfully recorded as time-series data for others to learn selection operators from.

A pure data recorder — it has no built-in learner and does not invert any fitness function. Phenotype is strictly a fixed function of sequence; "who is stronger" is never hand-written.

PyTorch backend + an Astro web visualizer ("the eye of acceptance"). The engine and its 68 primitives are in place (430 tests); two gates remain before data collection: a performance regression and a fixture re-record.
