# lookout

Tiered AI escalation for home camera feeds: a cheap, fast local detector watches
everything; a declarative chains file decides what deserves a closer look; a
priority scheduler decides who gets the scarce VLM inference time first; actions
fire into Home Assistant.

**Status: scaffold.** Quickstart, architecture notes, and the demo walkthrough
land with the build.

## Why

Frigate + Home Assistant already do single-shot "describe this event with a VLM."
What doesn't exist is the layer between: stateful multi-step condition chains
(vehicle seen → which vehicle → who got out) and a global priority queue over
constrained local inference, where a hand-gesture command (sub-second budget) must
jump ahead of driveway classification (a 20-second budget). lookout is that layer.
